"""
General RAG Pipeline Module as a Class

This module implements the overall retrieval-augmented generation (RAG) workflow
in a class-based approach. The RAGPipeline class encapsulates the state graph that
manages the following operations:
 - Retrieve relevant documents.
 - Grade document relevance.
 - Generate an answer.
 - Regenerate answers if hallucinations are detected.
 - Transform (rewrite) the query for improved retrieval.

Each node uses a separate prompt (imported from backend/prompts).
"""

import numpy as np
import logging
from langgraph.graph import START, END, StateGraph
from langchain_core.output_parsers import StrOutputParser
from together import Together
from langchain_groq import ChatGroq

# Import prompts and their associated output schemas.
from backend.general.index import create_faiss_index
from backend.general.ingestion import ingest_pipeline
from backend.general.prompts import (
    grade_documents_prompt, GradeDocuments,
    grade_hallucinations_prompt, GradeHallucinations,
    grade_answer_prompt, GradeAnswer,
    arxiv_prompt , ArxivCheck,
    web_search_prompt, WebSearchQuery,
    main_prompt, regeneration_prompt, rewriter_prompt
)
from config.settings import EMAILS_FILE

from typing import List
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a TypedDict for the graph state.
class GraphState(TypedDict):
    question: str
    transform_q: str
    prev_answer: str
    generation: str
    documents: List[str]
    num_reterival: int
    hallucination_try: int
    arxiv_query: None
    web_search_query : None
    
import getpass
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

class RAGPipeline:
    """
    A class that encapsulates the retrieval-augmented generation pipeline.
    This pipeline uses a state graph to manage document retrieval, grading,
    answer generation, regeneration, and query transformation.
    """
    def __init__(self):
        # Initialize LLM  objects.
        self.llm = ChatGroq(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",  # replace with a safe model
            temperature=0.0,
            max_retries=2,
            timeout=10
        )

        # Create structured LLM graders by combining the prompts with LLM structured output.
        structured_llm_grader_docs = self.llm.with_structured_output(GradeDocuments)
        self.retrieval_grader = grade_documents_prompt | structured_llm_grader_docs

        structured_llm_grader_hallucinations = self.llm.with_structured_output(GradeHallucinations)
        self.hallucination_grader = grade_hallucinations_prompt | structured_llm_grader_hallucinations

        structured_llm_grader_answer = self.llm.with_structured_output(GradeAnswer)
        self.answer_grader = grade_answer_prompt | structured_llm_grader_answer
        
        structured_llm_arxiv_check = self.llm.with_structured_output(ArxivCheck)
        self.arxiv_query = arxiv_prompt | structured_llm_arxiv_check
        
        structured_llm_web_search = self.llm.with_structured_output(WebSearchQuery)
        self.web_query = web_search_prompt | structured_llm_web_search
        
        # Prepare main generation and regeneration chains.
        self.rag_chain = main_prompt | self.llm | StrOutputParser()
        self.rag_re_chain = regeneration_prompt | self.llm | StrOutputParser()
        self.question_rewriter = rewriter_prompt | self.llm | StrOutputParser()

        # Compile the state graph workflow.
        self.workflow = StateGraph(GraphState)
        self.vectorstore , self.doc_splits = create_faiss_index(EMAILS_FILE) 
        self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self):
        """Define and add nodes and edges to the state graph."""
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("regenerate", self.regenerate)
        self.workflow.add_node("transform_query", self.transform_query)

        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges("grade_documents", self.decide_to_generate, {
            "transform_query": "transform_query",
            "generate": "generate",
        })
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_conditional_edges("generate", self.grade_generation_v_documents_and_question, {
            "not supported": "regenerate",
            "useful": END,
            "not useful": "transform_query",
        })
        self.workflow.add_conditional_edges("regenerate", self.grade_generation_v_documents_and_question, {
            "not supported": "regenerate",
            "useful": END,
            "not useful": "transform_query",
        })

    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve documents from the vector store via Together's embeddings."""
        question = state["transform_q"]
        results = self.vectorstore.search(question, search_type="similarity", k = 3)
        documents = []
        for i in range(3):
            documents.append(results[i])
            
        if state["arxiv_query"] or state["web_search_query"]:
            doc , vectordb = ingest_pipeline(state["arxiv_query"] , state["web_search_query"])
            if doc :
                for i in range(len(doc)):
                    documents.append(doc[i])
            doc = []
            
            if vectordb :
                results = vectordb.search(question , search_type = "similarity" , k = 2)
                for i in range(len(results)):
                    documents.append(results[i])
            
        state["documents"] = documents
        logger.info("Retrieved documents: %s", documents)
        return state

    def grade_documents(self, state: GraphState) -> GraphState:
        """Grade retrieved documents and filter out non-relevant ones."""
        question = state["transform_q"]
        num_reterival = state["num_reterival"]
        documents = state["documents"]
        filtered_docs = []
        
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score.lower()
            if grade == "yes":
                filtered_docs.append(d)
                
        if num_reterival >= 3 and not filtered_docs:
            return state
        state["documents"] = filtered_docs
        return state

    def generate(self, state: GraphState) -> GraphState:
        """Generate answer using the current documents and context."""
        question = state["question"]
        documents = state["documents"]
        state["generation"] = self.rag_chain.invoke({
            "context": documents,
            "prev_answer": state["prev_answer"],
            "question": question
        })
        return state

    def regenerate(self, state: GraphState) -> GraphState:
        """Generate a revised answer to address hallucinations."""
        question = state["question"]
        documents = state["documents"]
        state["hallucination_try"] += 1
        state["generation"] = self.rag_re_chain.invoke({
            "context": documents,
            "question": question,
            "prev_ans": state["prev_answer"],
            "ans": state["generation"]
        })
        return state

    def transform_query(self, state: GraphState) -> GraphState:
        """Rewrite the query to capture unresolved aspects from the previous answer."""
        question = state["question"]
        state["num_reterival"] += 1
        state["prev_answer"] = state["generation"]
        state["hallucination_try"] = 2
        better_question = self.question_rewriter.invoke({
            "question": question,
            "prev_answer": state["prev_answer"]
        })
        logger.info("Transformed query: %s", better_question)
        state["transform_q"] = better_question
        
        arxiv_check = self.arxiv_query.invoke({"query" : better_question})
        if arxiv_check.binary_score.lower() == 'yes' :
            state["arxiv_query"] = arxiv_check.improved_query 
            
        web_check = self.web_query.invoke({"query" : better_question})
        state["web_search_query"] = web_check.improved_query 
        return state

    def decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate an answer or re-transform the query."""
        filtered_documents = state["documents"]
        return "transform_query" if not filtered_documents else "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """
        Evaluate whether the generated answer is supported by context and if it resolves the question.
        Returns a string decision for the next node.
        """
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        num_reterival = state["num_reterival"]
        hallucination_try = state["hallucination_try"]
        prev_answer = state["prev_answer"]

        score = self.hallucination_grader.invoke({
            "documents": documents,
            "generation": generation,
            "prev_answer": prev_answer
        })
        grade = score.binary_score
        if grade.lower() == "yes" or hallucination_try >= 3:
            score = self.answer_grader.invoke({
                "question": question,
                "generation": generation
            })
            grade = score.binary_score.lower()
            if grade == "yes" or num_reterival >= 3:
                return "useful"
            else:
                logger.info("Answer not useful.\nGeneration: %s\nPrev Answer: %s", generation, prev_answer)
                return "not useful"
        else:
            logger.info("Generation not supported.\nPrev Answer: %s\nGeneration: %s", prev_answer, generation)
            return "not supported"

    def run(self, initial_state: GraphState) -> GraphState:
        """
        Execute the entire RAG pipeline workflow starting from the given initial state.
        
        Args:
            initial_state (GraphState): The starting state for the pipeline.
        
        Returns:
            GraphState: The final state after pipeline execution.
        """
        # The compiled state graph (self.app) is callable and accepts the initial state.
        for output in self.app.stream(initial_state):
            for key, value in output.items():
                # Node
                print(f"Node '{key}':")
            print("---")
        return value["generation"]


# For testing, you might run:
if __name__ == "__main__":
    # Create an example initial state.
    initial_state: GraphState = {
        "question": "Explain about aider.chat in the detail",
        "transform_q": "Explain about aider.chat in the detail",  # Initially the same
        "prev_answer": "",
        "generation": "",
        "documents": [],
        "num_reterival": 0,
        "hallucination_try": 0,
        "arxiv_query" : None,
        "web_search_query" : None
    }
    
    pipeline = RAGPipeline()
    final_state = pipeline.run(initial_state)
    print("Final State:", final_state)
