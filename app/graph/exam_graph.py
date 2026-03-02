"""
Exam reasoning graph: retrieve → answer → evaluate → retry (local or web fallback).
"""

import os
from typing import NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.weak_topics import record_weak_topics
from app.tools.web_search import get_web_tool


class ExamState(TypedDict):
    question: str
    answer: NotRequired[str]
    confidence: NotRequired[float]
    weak_topics: NotRequired[list]
    context: NotRequired[list]  # retrieved doc contents for answer_question
    retry_attempted: NotRequired[bool]  # True after one local retry
    used_web: NotRequired[bool]  # True after web fallback ran


def build_exam_graph(llm, retriever):
    """
    Build the exam reasoning graph.
    llm: LangChain LLM (e.g. ChatOpenAI).
    retriever: LangChain retriever (e.g. from Chroma.as_retriever()).
    """
    graph_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(graph_dir, "..", "prompts", "exam_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "You are an exam preparation assistant. Answer concisely with bullet points."

    def retrieve_context(state: ExamState):
        docs = retriever.invoke(state["question"])
        context = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
        return {**state, "context": context}

    def answer_question(state: ExamState):
        context = state.get("context") or []
        context_text = "\n\n".join(context)
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Context from materials:\n{context_text}\n\nQuestion: {state['question']}"
            ),
        ]
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {**state, "answer": answer}

    def evaluate_answer(state: ExamState):
        from langchain_core.messages import HumanMessage

        eval_prompt = (
            "Rate the following exam answer for correctness and completeness. "
            "Reply with exactly two lines: first line a number 0.0 to 1.0 (confidence), "
            "second line a comma-separated list of weak topics (or 'none')."
        )
        messages = [
            HumanMessage(
                content=f"Question: {state['question']}\nAnswer: {state['answer']}\n\n{eval_prompt}"
            ),
        ]
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        lines = text.strip().split("\n")
        try:
            confidence = float(lines[0].strip())
        except (ValueError, IndexError):
            confidence = 0.5
        weak_topics = []
        if len(lines) > 1 and "none" not in lines[1].lower():
            weak_topics = [t.strip() for t in lines[1].split(",") if t.strip()]
        # Persist weak topics for future practice / revision, especially when confidence is low.
        if weak_topics:
            record_weak_topics(weak_topics, confidence)
        return {**state, "confidence": confidence, "weak_topics": weak_topics}

    def retry_if_low_confidence(state: ExamState):
        # Retrieve more context from vectorstore (weak topics to broaden search)
        query = state["question"]
        if state.get("weak_topics"):
            query = f"{query} {' '.join(state['weak_topics'])}"
        docs = retriever.invoke(query)
        context = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
        return {**state, "context": context, "retry_attempted": True}

    web_tool = get_web_tool()

    def web_search_fallback(state: ExamState):
        # Fallback: use DuckDuckGo when local context wasn't enough (LangChain tool in LangGraph)
        query = state["question"]
        if state.get("weak_topics"):
            query = f"{query} {' '.join(state['weak_topics'])}"
        result = web_tool.invoke(query) if hasattr(web_tool, "invoke") else web_tool.run(query)
        context = list(state.get("context") or [])
        context.append(f"[Web search for: {query}]\n{result}")
        return {**state, "context": context, "used_web": True}

    def route_after_evaluate(state: ExamState):
        if state.get("confidence", 0) >= 0.7:
            return "end"
        if state.get("used_web"):
            return "end"  # already used web; don't loop again
        if state.get("retry_attempted"):
            return "web"  # tried local retry once; fall back to web
        return "retry"  # first time low confidence: retry with more local context

    builder = StateGraph(ExamState)

    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("answer_question", answer_question)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("retry_if_low_confidence", retry_if_low_confidence)
    builder.add_node("web_search_fallback", web_search_fallback)

    builder.add_edge("retrieve_context", "answer_question")
    builder.add_edge("answer_question", "evaluate_answer")
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluate,
        {
            "end": END,
            "retry": "retry_if_low_confidence",
            "web": "web_search_fallback",
        },
    )
    builder.add_edge("retry_if_low_confidence", "answer_question")
    builder.add_edge("web_search_fallback", "answer_question")

    builder.set_entry_point("retrieve_context")

    return builder.compile()
