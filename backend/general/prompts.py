from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

system = (
    "You are a grader evaluating the relevance of a retrieved document to a user question. "
    "Assess whether the document contains relevant keywords or semantics that relate to any part of the user question. "
    "Provide a binary score: 'yes' if relevant, 'no' if not."
)
grade_documents_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document:\n{document}\nQuestion:\n{question}")
])


class GradeHallucinations(BaseModel):
    """Binary score for hallucination presence in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

system = (
    "Evaluate if the LLM response is supported by the provided context and any previous answer (if available).\n"
    "Provide a binary score: 'yes' if the response is supported; 'no' if it lacks support. Ignore disclaimers."
)
grade_hallucinations_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Context:\n{documents}\nPrevious Answer:\n{prev_answer}\nLLM Generation:\n{generation}")
])

class GradeAnswer(BaseModel):
    """Binary score to assess if an answer addresses the given question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

system = (
    "You are tasked with evaluating whether an answer resolves the given question.\n"
    "Provide a binary score: 'yes' if the answer resolves the question, 'no' otherwise."
)
grade_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "User question:\n{question}\nLLM Generation:\n{generation}")
])


system = (
    "You are an assistant for question-answering tasks. Use the following retrieved CONTEXT and "
    "PREVIOUS ANSWER (if given) to formulate your response. Ensure your answer is strictly supported "
    "by the context. If a part cannot be answered with the available info, state that more information is needed."
)
main_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Context:\n{context}\nPrevious Answer:\n{prev_answer}\nQuestion:\n{question}\nAnswer:")
])

system = (
    "You're an assistant for question-answering tasks. Given a question, context, a previous incomplete answer, "
    "and a current answer with hallucinations, use the context and previous answer to correct inaccuracies. "
    "Answer only parts fully supported by the context or previous answer. For unsupported parts, state that more information is needed. "
    "Keep your response concise and aligned with the context."
)
regeneration_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Question:\n{question}\nPrevious Answer:\n{prev_ans}\nContext:\n{context}\nAnswer:\n{ans}")
])

system = (
    "You are a question re-writer that reformulates input questions for optimal vectorstore retrieval. "
    "Review the initial question and the previous answer; focus on unresolved aspects. "
    "Output only the revised question, optimized for retrieval without additional commentary."
)
rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Here is the initial question:\n{question}\nPrevious Answer:\n{prev_answer}\n\nGenerate a refined question:")
])

class ArxivCheck(BaseModel):
    """Binary score indicating whether arXiv retrieval would help, plus an improved query."""
    binary_score: str = Field(description="Whether external research papers are needed: 'yes' or 'no'")
    improved_query: str = Field(description="Optimized query to retrieve relevant papers from arXiv if needed.")
    
system_message = (
    "You are an academic assistant. Determine if external research papers from arXiv would help answer the query.\n"
    "If yes, rewrite the query to be more effective for retrieving relevant papers using arXiv's API.\n"
    "If no, still return an improved version of the question for general search.\n"
    "Respond only with a binary score and the improved query."
)

arxiv_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "User Query:\n{query}")
])

class WebSearchQuery(BaseModel):
    """An improved query string optimized for web search."""
    improved_query: str = Field(description="Optimized version of the original query for better web search results.")
    
web_search_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an assistant optimizing user queries for web search.\n"
     "Rewrite the query to be more specific, informative, and suitable for web search engines.\n"
     "Avoid vague terms. Include useful keywords, context, or recent-year filters if relevant."),
    ("human", "Query:\n{query}")
])