
from PrototipoRAG import obtener_mejor_chunk
import json
from datasets import Dataset

def load_questions_jsonl(path="evaluacion/questions.json"):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line)["question"])
    return questions

def build_eval_dataset(questions, model="llama3.1:8b-instruct-q4_K_M"):
    rows = {"question": [], "answer": [], "contexts": []}

    for q in questions:
        r = obtener_mejor_chunk(q, model=model)
        contexts = [x["chunk"] for x in (r.get("retrieved") or [])]

        rows["question"].append(q)
        rows["answer"].append(r.get("answer", ""))
        rows["contexts"].append(contexts)

    return Dataset.from_dict(rows)

if __name__ == "__main__":
    while True:
        pregunta = input("Pregunta (enter para salir): ").strip()
        if not pregunta:
            break

        result = obtener_mejor_chunk(pregunta, model="llama3.1:8b-instruct-q4_K_M")

        print("\n=== Respuesta del LLM ===")
        print(result["answer"])
        print("\n=== Metadatos del chunk ===")
        print("Título:", result["title"])
        print("Fichero:", result["filename"])
        print("Segmento:", result["segment_index"])
        print("\n=== Chunk usado ===")
        print(result["chunk"])
        print("\n" + "=" * 60 + "\n")
        
        
