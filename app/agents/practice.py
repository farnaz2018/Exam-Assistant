"""
Generate practice questions from documents and evaluate user answers.
"""

import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.weak_topics import top_weak_topics


def _prompts_dir():
    return os.path.join(os.path.dirname(__file__), "..", "prompts")


def _load_practice_prompt():
    path = os.path.join(_prompts_dir(), "practice_questions_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "Generate 5 exam-style questions from this content. "
            "Do not show answers. Use format: QUESTION N: ... EXPECTED_ANSWER N: ..."
        )


def _parse_qa_blocks(text: str) -> list[dict[str, str]]:
    """Parse QUESTION N: ... EXPECTED_ANSWER N: ... blocks into list of {question, expected_answer}."""
    qa_list = []
    # Match QUESTION 1: ... EXPECTED_ANSWER 1: ... (allow multiline)
    pattern = re.compile(
        r"QUESTION\s+(\d+):\s*(.*?)\s*EXPECTED_ANSWER\s+\1:\s*(.*?)"
        r"(?=QUESTION\s+\d+:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        q = m.group(2).strip()
        a = m.group(3).strip()
        if q:
            qa_list.append({"question": q, "expected_answer": a or "(no answer)"})
    # Fallback: split by EXPECTED_ANSWER to get Q and A pairs
    if not qa_list and "QUESTION" in text.upper():
        blocks = re.split(r"\s*QUESTION\s+\d+:\s*", text, flags=re.IGNORECASE)
        for block in blocks[1:6]:
            if "EXPECTED_ANSWER" in block.upper():
                parts = re.split(r"\s*EXPECTED_ANSWER\s+\d+:\s*", block, maxsplit=1, flags=re.IGNORECASE)
                q = (parts[0].strip() if parts else "").strip()
                a = (parts[1].strip() if len(parts) > 1 else "").strip() or "(no answer)"
                if q:
                    qa_list.append({"question": q, "expected_answer": a})
            else:
                q = block.strip()
                if q:
                    qa_list.append({"question": q, "expected_answer": "(no answer)"})
    return qa_list[:5]


def generate_practice_test_from_retriever(
    llm: Any, retriever: Any, topic: str | None = None, k: int = 8
) -> dict[str, Any]:
    """
    Generate a practice test from retrieved document chunks.
    topic: optional query to retrieve relevant content; if None, bias toward weak topics.
    """
    if topic:
        query = topic
    else:
        weak = top_weak_topics()
        if weak:
            # Prioritize lowest-confidence topics first.
            query = " ".join(weak[:3])
        else:
            query = "main concepts, definitions, and exam topics"
    docs = retriever.invoke(query) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(query)
    chunks = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    content = "\n\n".join(chunks[:k])
    if not content.strip():
        content = "No document content available. Add PDF/DOCX/PNG files to data/raw and run ingestion."
    return generate_practice_test(llm, content)


def generate_practice_test(llm: Any, content: str) -> dict[str, Any]:
    """
    Generate 5 exam-style questions from the given document content.
    Returns a practice test: questions to show (no answers) and internal Q&A for grading.
    """
    instruction = _load_practice_prompt()
    messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=f"Content:\n\n{content}"),
    ]
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    qa_list = _parse_qa_blocks(text)
    if not qa_list:
        qa_list = [
            {"question": f"Question {i+1} (parse failed)", "expected_answer": "N/A"}
            for i in range(5)
        ]
    questions_display = [item["question"] for item in qa_list]
    return {
        "questions_display": questions_display,
        "questions_with_answers": qa_list,
    }


def evaluate_answers(
    llm: Any,
    questions_with_answers: list[dict[str, str]],
    user_answers: list[str],
) -> dict[str, Any]:
    """
    Grade user answers and return score plus feedback.
    questions_with_answers: list of {"question", "expected_answer"} from generate_practice_test.
    user_answers: list of strings, one per question (same order).
    """
    grading_prompt = """You are grading exam answers. For each question:
- Compare the user's answer to the expected answer.
- Say if it is correct, partial, or wrong.
- Give a short explanation and highlight what was missed if wrong.

Output format:
For each of the 5 questions, output a block:
QUESTION N: <repeat question briefly>
USER ANSWER: <user's answer>
GRADE: correct | partial | wrong
SCORE: <0 or 0.5 or 1>
EXPLANATION: <one or two sentences>

Then at the end output one line:
TOTAL_SCORE: <sum of the 5 scores, so 0-5>
"""
    parts = []
    for i, qa in enumerate(questions_with_answers):
        u = user_answers[i] if i < len(user_answers) else "(no answer)"
        parts.append(
            f"Q{i+1}. {qa['question']}\nExpected: {qa['expected_answer']}\nUser answer: {u}"
        )
    body = "\n\n".join(parts)
    messages = [
        SystemMessage(content=grading_prompt),
        HumanMessage(content=body),
    ]
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    # Parse TOTAL_SCORE
    total_match = re.search(r"TOTAL_SCORE:\s*([\d.]+)", text, re.IGNORECASE)
    raw_score = float(total_match.group(1)) if total_match else 0.0
    score_pct = min(100, max(0, (raw_score / 5.0) * 100))
    return {
        "score": round(score_pct, 1),
        "feedback": text,
        "raw_score": raw_score,
    }
