"""
code_prompts.py

This module defines the prompt templates used by the Graph RAG system for the codebase chatbot.

Prompts Included:
- Query Rewriter Prompt: Reformulates the user query to optimize retrieval.
- Code Reasoning Prompt: Generates a code-based answer based on the provided context.

Author: Your Name
Date: YYYY-MM-DD
"""

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ------------------ Query Rewriter Prompt ------------------

# System prompt instructs the LLM to rewrite the query for better code retrieval.
class QueryReWrite(BaseModel):
    improved_query: str = Field(
        ..., 
        description="An optimized version of the original query for efficient retrieval."
    )

rewriter_system_prompt = (
    "You are a question rewriter specialized in code retrieval. Your task is to examine "
    "an input question and rephrase it so that the essential code-related elements are emphasized. "
    "If the query is ambiguous or broad, narrow it down to focus on function names, class names, "
    "or specific code constructs. Return only the revised query without additional commentary."
)

rewriter_human_prompt = (
    "Initial Question:\n{question}\n\nRewritten Query:"
)

query_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", rewriter_system_prompt),
    ("human", rewriter_human_prompt)
])


# ------------------ Code Reasoning Prompt ------------------

# System prompt instructs the LLM to generate a code answer using the provided context.
code_reasoning_system_prompt = (
    "You are a highly skilled code assistant. Using only the provided code context, answer the following query precisely. "
    "Base your answer strictly on the context and refer to specific code elements as needed. "
    "If the provided context does not cover a detail required for the answer, explicitly state that the information is unavailable. "
    "Keep your response concise and focused on the code."
)