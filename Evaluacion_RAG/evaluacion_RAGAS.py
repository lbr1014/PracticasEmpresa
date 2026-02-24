import json
from pathlib import Path

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


# LLM judge
from langchain_ollama import ChatOllama

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


from PrototipoRAG import obtener_mejor_chunk 

def load_questions(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in data:
        if isinstance(item, str):
            out.append({"question": item})
        else:
            out.append(item)
    return out


def build_ragas_rows(questions: list[dict], k_contexts: int = 5) -> list[dict]:
    rows = []
    for q in questions:
        question = (q.get("question") or "").strip()
        if not question:
            continue

        rag_out = obtener_mejor_chunk(question)  
        answer = (rag_out.get("answer") or "").strip()

        retrieved = rag_out.get("retrieved") or []
        contexts = []
        for r in retrieved[:k_contexts]:
            chunk = (r.get("chunk") or "").strip()
            if chunk:
                contexts.append(chunk)

        row = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }

        if "ground_truth" in q and q["ground_truth"]:
            row["ground_truth"] = q["ground_truth"]

        rows.append(row)

    return rows


def main():
    questions_path = Path("evaluacion/questions.json") 
    questions = load_questions(questions_path)

    rows = build_ragas_rows(questions, k_contexts=5)
    ds = Dataset.from_list(rows)

    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0, base_url="http://localhost:11434")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print(result)
    df = result.to_pandas()
    print(df.to_string(index=False))



if __name__ == "__main__":
    main()
