"""
Code Graph Builder

This module builds a scalable code graph using NetworkX. It analyzes
a codebase (e.g., Python files) to extract code entities and their relationships.
Relationships include:
    - Imports between modules
    - Inheritance between classes
    - Method calls and constructor usage
    - Additional relationships (as needed)

Nodes represent code entities (classes, functions, methods, etc.) and edges
capture the relationships between these entities. This data structure can be used
for a Graph RAG pipeline or further code analysis.
"""

from collections import defaultdict
import os
import logging
from matplotlib import pyplot as plt
import networkx as nx

# Import Tree-sitter components if available.
import torch.nn.functional as F
import torch
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser
from backend.codebase.utils import LanguageEnum 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from together import Together
from langchain_groq import ChatGroq
from backend.codebase.prompt import (QueryReWrite, query_rewriter_prompt , code_reasoning_system_prompt)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
CODING_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"
# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a mapping from file extensions to Tree-sitter language names.
# Define your BLACKLIST_DIR, WHITELIST_FILES, NODE_TYPES, and REFERENCE_IDENTIFIERS here
BLACKLIST_DIR = [
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".git",
    ".idea",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".vscode",
    ".github",
    ".gitlab",
    ".angular",
    "cdk.out",
    ".aws-sam",
    ".terraform"
]
WHITELIST_FILES = [".java", ".py", ".js", ".rs"]
BLACKLIST_FILES = ["docker-compose.yml"]

def get_language_from_extension(file_ext):
    """
    Just temprory
    """
    FILE_EXTENSION_LANGUAGE_MAP = {
        ".java": LanguageEnum.JAVA,
        ".py": LanguageEnum.PYTHON,
        ".js": LanguageEnum.JAVASCRIPT,
        ".rs": LanguageEnum.RUST,
        # Add other extensions and languages as needed
    }
    return FILE_EXTENSION_LANGUAGE_MAP.get(file_ext)

class CodeGraphBuilder:
    """
    A class to build a code graph for a given codebase.

    This builder:
      - Recursively lists code files with given extensions.
      - Parses each file using Tree-sitter (if available).
      - Extracts code entities and relationships (e.g., imports, inheritance).
      - Populates a NetworkX graph with nodes (code entities) and edges (relationships).
    """
    
    def __init__(self, code_root: str, extensions: list = None):
        """
        Initialize the CodeGraphBuilder.

        Args:
            code_root (str): The root directory of the codebase.
            extensions (list, optional): List of file extensions to consider (e.g., ['.py', '.rs']).
                                         Defaults to ['.py'].
        """
        self.code_root = code_root
        self.extensions = extensions if extensions is not None else ['.py']
        
        # Initialize the NetworkX graph.
        self.graph = nx.DiGraph()
        
        self.client = Together()
        self.llm = ChatGroq(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",  # replace with a safe model
            temperature=0.0,
            max_retries=2,
            timeout=10
        )
        
        structured_llm_query_writer = self.llm.with_structured_output(QueryReWrite)
        self.query_rewrite = query_rewriter_prompt | structured_llm_query_writer
    
    def list_code_files(self) -> list:
        """
        Recursively list all code files in the codebase with the given extensions.
        
        Returns:
            list: List of absolute file paths.
        """
        code_files = []
        for root, dirs, files in os.walk(self.code_root):
            dirs[:] = [d for d in dirs if d not in BLACKLIST_DIR]
            for file in files:
                file_ext = os.path.splitext(file)[1]
                if file_ext in self.extensions:
                    if file not in BLACKLIST_FILES:
                        file_path = os.path.join(root, file)
                        language = get_language_from_extension(file_ext)
                        if language:
                            code_files.append(file_path)
                            
        logger.info(f"Found {len(code_files)} code files in {self.code_root}.")
        return code_files

    def parse_file(self, file_path: str):
        """
        Parse a single file using Tree-sitter if available; otherwise, return the raw content.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            tuple: (root_node, file_content) if parsed successfully, or (None, content) otherwise.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None, None
        
        ext = os.path.splitext(file_path)[1]
        if ext in self.extensions and Parser is not None:
            try:
                parser = Parser()
                parser.set_language(get_language(get_language_from_extension(ext).value))
                tree = parser.parse(bytes(content, "utf8"))
                return tree.root_node, content
            except Exception as e:
                logger.error(f"Tree-sitter parsing failed for {file_path}: {e}")
                return None, content
        else:
            # Fallback: no parsing; simply return the content.
            return None, content
    
    def extract_entities(self, node, content, file_path) :
        """
        Extract code entities from the parse tree.

        Args:
            node: The root node of the parse tree.
            content (str): The source code as a string.
        """
        
        for child in node.children:
            # Check for a class definition (using the query approach is preferable).
            if child.type in ["class_definition", "class"]:
                self._process_class_statement(child, file_path)
            elif child.type in ["function_definition", "function"]:
                self._process_function_statement(child, file_path)
            
            # Can be extended to handle import and expression as well    
            
            # elif child.type == ["import_from_statement" , "import_statement"]:
            #     self._process_import_statement(node, file_path)
            # elif child.type == "expression_statement":
            #     self._process_expression_statement(node, file_path)

    def _process_function_statement(self, node, file: str, class_name = None):
        """
        Processes a function definition node within a class (method).
        
        Args:
            node: The Tree-sitter node representing the method.
            parent_class (str): The name of the parent class.
            file (str): The filename where this method is defined.
            G (nx.DiGraph): The NetworkX graph.
        """
        func_name_node = node.child_by_field_name("name")
        func_name = func_name_node.text.decode("utf8")
        
        # Extracts function parameters
        params = []
        param_node = node.child_by_field_name("parameters")
        if param_node:
            params = [p.text.decode("utf8") for p in param_node.children if p.type == "identifier"]
        
        # Extracts the return type
        return_node = node.child_by_field_name("return_type")
        ret = return_node.text.decode("utf8") if return_node else "Unknown"
        
        # Extracts the docstring
        docstring = ""
        suite = node.child_by_field_name('body')
        if suite and suite.type == 'block' and len(suite.children) > 0:
            first_stmt = suite.children[0]
            if first_stmt.type == 'expression_statement' and len(first_stmt.children) > 0 and first_stmt.children[0].type == 'string' :
                docstring = first_stmt.text.decode("utf8").strip()
        
        # Create a node for the method.
        self.graph.add_node(func_name, type="function", name=func_name, class_name = class_name, file=file, params=params, ret=ret,  docstring=docstring, code = node.text.decode("utf8"))
        self.graph.add_edge(file, func_name, relation="contain")
        
        if class_name : 
            self.graph.add_edge(class_name, func_name, relation="contain method")
            
        
        logger.debug(f"Processed method: {func_name} in file {file}")
    
    def _process_class_statement(self, node, file: str):
        """
        Processes a class definition node and adds it to the graph.
        
        Args:
            node: The Tree-sitter node for the class definition.
            file (str): The filename where the class is defined.
        """
        # Get the name of the class.
        class_name_node = node.child_by_field_name("name")
        class_name = class_name_node.text.decode("utf8")
        
        # Extracts the docstring
        docstring = ""
        suite = node.child_by_field_name('body')
        if suite and suite.type == 'block' and len(suite.children) > 0:
            first_stmt = suite.children[0]
            if first_stmt.type == 'expression_statement' and len(first_stmt.children) > 0 and first_stmt.children[0].type == 'string' :
                docstring = first_stmt.text.decode("utf8").strip()
                
            
        # Count methods: count children of type "function_definition".
        num_methods = sum(1 for n in suite.children if n.type == "function_definition")
        
        inheritance = None
        supercalss = node.child_by_field_name("superclass")
        if supercalss : 
            inheritance = supercalss.text.decode("utf8") 
            
        
        args = ""
        for stmt in suite.children:
            if stmt.type == 'function_definition':
                func_name = stmt.child_by_field_name('name').text.decode('utf8')
                if func_name == '__init__':
                    param_node = stmt.child_by_field_name('parameters')
                    if param_node:
                        args = []
                        for param in param_node.children:
                            if param.type in ('identifier', 'default_parameter', 'typed_parameter'):
                                args.append(param.text.decode('utf8'))
                            elif param.type in ('*', '**', '*args', '**kwargs'):
                                args.append(param.text.decode('utf8'))
                            elif param.type == ',':
                                continue

        # Add an edge from the file to the class.
        self.graph.add_node(class_name, type="class", name=class_name, file=file,
                docstring=docstring, number_of_methods=num_methods, inheritance=inheritance, params = args, code = "")
        self.graph.add_edge(file, class_name, relation="contains")
        logger.debug(f"Processed class: {class_name}")

        # Process methods within the class.
        for class_child in node.children:
            # Typically, methods are within a block node.
            if class_child.type == "block":
                for method in class_child.children:
                    if method.type == "function_definition":
                        self._process_function_statement(method, file, class_name)

    def get_node_text(self, n, node):
        if node['type'] == "function":
            return f"{node["file"]} \n{node['name']}({node['params']}) -> {node['ret']}\n{node['docstring'] or ''}"
        if node['type'] == "class":
            return f"{node["file"]} \n{node['name']}({node['inheritance']}) -> {node['docstring']}\n def __init__({node["params"]}): \nNumber of methods {node['number_of_methods'] or ''}"
    
    def build_graph(self) -> nx.DiGraph:
        """
        Build the code graph by processing all code files.

        Nodes represent code entities.
        Edges represent relationships (e.g., imports, inheritance). For this basic example,
        no relationships are added; this is a placeholder for you to extend.

        Returns:
            nx.DiGraph: A NetworkX directed graph of code entities.
        """
        files = self.list_code_files()
            
        for file_path in files:
            root_node, content = self.parse_file(file_path)
            if root_node is not None:
                self.extract_entities(root_node, content, file_path)

        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        self.create_embedding()
        return self.graph

    def visualize_graph(self, output_file = None) -> None:
        """
        Visualize the provided NetworkX code graph using Matplotlib.

        Nodes are colored based on their 'type' attribute.
        If an output_file path is provided, the plot is saved to that file;
        otherwise, the plot is displayed interactively.

        Args:
            G (nx.DiGraph): The NetworkX directed graph to visualize.
            output_file (str, optional): Path to save the output image (e.g., "graph.png"). Defaults to None.
        """
        # Compute positions for the nodes using spring layout.
        pos = nx.spring_layout(self.graph, k=0.5, seed=42)
        
        # Define colors for different node types.
        type_to_color = {
            "file": "skyblue",
            "class": "lightgreen",
            "function": "salmon",
            "module": "violet",
            "variable": "orange"
        }
        
        # Prepare node colors based on the 'type' attribute. Default to gray.
        node_colors = []
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "default")
            color = type_to_color.get(node_type, "gray")
            node_colors.append(color)
        
        # Create the plot.
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors,
                node_size=500, font_size=8, arrows=True)
        
        # Save or show the plot.
        if output_file:
            plt.savefig(output_file)
            print(f"Graph visualization saved to {output_file}")
        else:
            plt.show()

    def create_embedding(self):
        for n, data in self.graph.nodes(data=True):
            if n and data and len(data.keys()) > 0:
                data['embedding'] = embedding_model.embed_query(self.get_node_text(n , data))
    
    def retrieve(self, query):
        query_embedding = torch.Tensor(embedding_model.embed_query(query))
        personalization = {
            n: F.cosine_similarity(query_embedding, torch.Tensor(data["embedding"]), dim=0).item()
            for n, data in self.graph.nodes(data=True)
            if 'embedding' in data
        }

        
        def get_code_for_node(n):
            data = self.graph.nodes[n]
            return str(data.get("doctsring")) + data.get('code') 
        
        ppr_scores = nx.pagerank(graph, personalization=personalization, alpha=0.85)
        top_functions = sorted([(n, s) for n, s in ppr_scores.items() if 'type' in self.graph.nodes[n].keys()  and self.graph.nodes[n]['type'] == 'function'],key=lambda x: -x[1])
        retrieved_code = "\n\n".join([f"# Function: {self.graph.nodes[n]['name']}\n{get_code_for_node(n)}" for n, _ in top_functions[:5]])
        
        return retrieved_code
    
    def answer(self , query):
        revised_query = self.query_rewrite.invoke({"question": query}).improved_query
        context = self.retrieve(revised_query)
        
        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            messages=[{"system" : code_reasoning_system_prompt,  "role":"user","content":f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"}])
        
        answer = response.choices[0].message.content
        return answer
        

if __name__ == "__main__":
    # For testing: provide the root directory of your code.
    # Example usage: python ingestion_codebase.py ./my_codebase_directory
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion_codebase.py <codebase_root_directory>")
        sys.exit(1)
    
    code_root_directory = sys.argv[1]
    builder = CodeGraphBuilder(code_root=code_root_directory, extensions=['.py', '.rs'])
    graph = builder.build_graph()
    
    # For demonstration, print some nodes and edges.
    print("Nodes in graph:")
    for node in list(graph.nodes)[:5]:
        print(node, graph.nodes[node])
    
    print("\nEdges in graph:")
    for edge in list(graph.edges)[:]:
        print(edge, graph.edges[edge])
    
    builder.visualize_graph()
    docs = builder.answer("Write fuction for parsing the file using treesitter")
    print(docs)
