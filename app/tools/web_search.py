"""
Web search tool for the exam assistant. Use as fallback when documents are insufficient.
"""

from langchain_community.tools import DuckDuckGoSearchRun


def get_web_tool():
    return DuckDuckGoSearchRun()
