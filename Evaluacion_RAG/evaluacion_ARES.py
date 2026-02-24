from ares import ARES

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Importa tu RAG
from PrototipoRAG import obtener_mejor_chunk


# Carga de preguntas
def load_questions(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, str):
            out.append({"question": item})
        elif isinstance(item, dict):
            out.append(item)
        else:
            continue
    return out


# Ejecutar el RAG y construir filas para ARES
def build_ares_rows(questions: list[dict[str, Any]], k_contexts: int = 5) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for q in questions:
        question = (q.get("question") or "").strip()
        if not question:
            continue

        rag_out = obtener_mejor_chunk(question) or {}
        answer = (rag_out.get("answer") or "").strip()

        retrieved = rag_out.get("retrieved") or []
        contexts: list[str] = []
        for r in retrieved[:k_contexts]:
            chunk = (r.get("chunk") or "").strip()
            if chunk:
                contexts.append(chunk)

        document = "\n\n".join(contexts).strip()

        rows.append(
            {
                "Query": question,
                "Document": document,
                "Answer": answer,
            }
        )

    return rows


# Guardar el TSV "unlabeled"
def write_tsv(rows: list[dict[str, str]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Query", "Document", "Answer"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (r.get(k) or "") for k in fieldnames})


# Ejecutar ARES (UES/IDP)
def run_ares_ues_idp(
    unlabeled_tsv: str | Path,
    few_shot_tsv: str | Path,
    model_choice: str,
    vllm: bool = False,
    host_url: str | None = None,
) -> Any:

    ues_idp_config: dict[str, Any] = {
        "in_domain_prompts_dataset": str(few_shot_tsv),
        "unlabeled_evaluation_set": str(unlabeled_tsv),
        "model_choice": model_choice,
    }

    # Ejecución local 
    if vllm:
        ues_idp_config["vllm"] = True
        if host_url:
            ues_idp_config["host_url"] = host_url

    ares = ARES(ues_idp=ues_idp_config)
    results = ares.ues_idp()
    return results


# Main: CLI y pipeline completo
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación RAG con ARES (generación TSV + UES/IDP opcional).")
    parser.add_argument(
        "--questions_path",
        type=str,
        default="evaluacion/questions.json",
        help="Ruta al JSON de preguntas (por defecto: evaluacion/questions.json).",
    )
    parser.add_argument(
        "--out_tsv",
        type=str,
        default="ares_unlabeled_output.tsv",
        help="Ruta de salida para el TSV unlabeled (por defecto: ares_unlabeled_output.tsv).",
    )
    parser.add_argument(
        "--k_contexts",
        type=int,
        default=5,
        help="Número de chunks recuperados a concatenar en Document (por defecto: 5).",
    )

    # Flags para ejecutar ARES
    parser.add_argument(
        "--run_ares",
        action="store_true",
        help="Si se indica, ejecuta ARES en modo UES/IDP después de generar el TSV.",
    )
    parser.add_argument(
        "--few_shot_tsv",
        type=str,
        default="evaluacion/ares_few_shot_prompt.tsv",
        help="Ruta al TSV few-shot con etiquetas para el judge (requerido si --run_ares).",
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="Modelo juez para ARES (por defecto: gpt-3.5-turbo-0125).",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Usa un endpoint OpenAI-compatible (vLLM) para ejecutar el modelo juez localmente.",
    )
    parser.add_argument(
        "--host_url",
        type=str,
        default=None,
        help='URL del endpoint OpenAI-compatible (ej: "http://0.0.0.0:8000/v1"). Solo si --vllm.',
    )

    args = parser.parse_args()

    questions_path = Path(args.questions_path)

    # Cargar preguntas
    questions = load_questions(questions_path)

    # Ejecutar RAG y construir filas (Query, Document, Answer)
    rows = build_ares_rows(questions, k_contexts=args.k_contexts)

    # Escribir TSV unlabeled
    out_tsv = Path(args.out_tsv)
    write_tsv(rows, out_tsv)
    print(f"TSV unlabeled generado: {out_tsv.resolve()} (filas: {len(rows)})")

    # Ejecutar ARES UES/IDP
    if args.run_ares:
        few_shot = Path(args.few_shot_tsv)
        if not few_shot.exists():
            print(
                "ERROR: --run_ares requiere un TSV few-shot existente.\n"
                f"No encuentro: {few_shot}\n\n"
                "Crea un TSV con cabecera:\n"
                "Query<TAB>Document<TAB>Answer<TAB>Context_Relevance_Label<TAB>Answer_Relevance_Label<TAB>Answer_Faithfulness_Label\n"
                "y ejemplos etiquetados con [[Yes]] / [[No]].",
                file=sys.stderr,
            )
            return 2

        results = run_ares_ues_idp(
            unlabeled_tsv=out_tsv,
            few_shot_tsv=few_shot,
            model_choice=args.model_choice,
            vllm=args.vllm,
            host_url=args.host_url,
        )

        print("\n=== Resultados ARES (UES/IDP) ===")
        try:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        except Exception:
            print(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
