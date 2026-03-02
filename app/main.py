"""
Entry point for the exam assistant application.
Supports: ingest, ask (graph), practice (generate + evaluate).
"""

import argparse
import os
import sys

# Add project root so "app" package resolves when run as script
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# Load .env (project root or app/ so OPENAI_API_KEY is set)
from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, ".env"))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def _get_llm():
    from app.config import get_llm as _get_llm_from_config
    return _get_llm_from_config()


def _get_retriever():
    from app.config import get_embeddings
    from langchain_community.vectorstores import Chroma
    persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectorstore")
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run: python -m app.main ingest"
        )
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def cmd_ingest(args):
    from app.ingest.ingest import create_vector_store, ingest_file
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    if not os.path.isdir(raw_dir):
        print("data/raw not found. Create it and add PDF/DOCX/PNG files.")
        return
    all_docs = []
    for root, _dirs, files in os.walk(raw_dir):
        for f in files:
            path = os.path.join(root, f)
            if path.endswith((".pdf", ".docx", ".png")):
                try:
                    docs = ingest_file(path)
                    all_docs.extend(docs)
                    print(f"Ingested: {path}")
                except Exception as e:
                    print(f"Skip {path}: {e}")
    if not all_docs:
        print("No documents loaded. Add PDF/DOCX/PNG under data/raw.")
        return
    create_vector_store(all_docs)
    print("Vector store saved to data/vectorstore.")


def cmd_ask(args):
    from app.graph.exam_graph import build_exam_graph
    llm = _get_llm()
    retriever = _get_retriever()
    graph = build_exam_graph(llm, retriever)
    out = graph.invoke({"question": args.question})
    print(out.get("answer", ""))
    if out.get("confidence") is not None:
        print(f"\nConfidence: {out['confidence']}")


def cmd_practice(args):
    from app.agents.practice import generate_practice_test_from_retriever
    llm = _get_llm()
    retriever = _get_retriever()
    result = generate_practice_test_from_retriever(llm, retriever, topic=args.topic)
    questions = result["questions_display"]
    print("Practice test (5 questions). Do not look up answers.\n")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}\n")
    # Store for evaluate: save path or in-memory for same process
    state_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "last_practice.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    import json
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(result["questions_with_answers"], f, indent=2)
    print("Answers saved for grading. Run: python -m app.main evaluate < your_answers.txt")
    print("Or: python -m app.main evaluate --answers 'ans1' 'ans2' 'ans3' 'ans4' 'ans5'")


def cmd_evaluate(args):
    from app.agents.practice import evaluate_answers
    import json
    state_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "last_practice.json")
    if not os.path.isfile(state_path):
        print("No practice test found. Run: python -m app.main practice")
        return
    with open(state_path, "r", encoding="utf-8") as f:
        qa = json.load(f)
    if args.answers:
        user_answers = list(args.answers)
    else:
        lines = sys.stdin.read().strip().split("\n")
        user_answers = [ln.strip() for ln in lines if ln.strip()][:5]
    while len(user_answers) < len(qa):
        user_answers.append("(no answer)")
    llm = _get_llm()
    result = evaluate_answers(llm, qa, user_answers)
    print(f"Score: {result['score']}%")
    print("\nFeedback:\n" + result["feedback"])


def main():
    parser = argparse.ArgumentParser(description="Exam assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Ingest data/raw into vector store")
    q = sub.add_parser("ask", help="Ask a question (exam graph)")
    q.add_argument("question", nargs="+", help="Your question (words)")
    p = sub.add_parser("practice", help="Generate 5 practice questions from documents")
    p.add_argument("--topic", default=None, help="Optional topic for retrieval")
    e = sub.add_parser("evaluate", help="Grade last practice answers (--answers or stdin)")
    e.add_argument("--answers", nargs="*", default=None, help="Five answers in order")
    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        args.question = " ".join(args.question)
        cmd_ask(args)
    elif args.command == "practice":
        cmd_practice(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
