"""
Exam-focused RetrievalQA agent.
"""

try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
except ImportError:
    from langchain.chains import RetrievalQA


def build_exam_agent(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )
