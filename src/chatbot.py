import argparse
import ollama
from abc import ABC, abstractmethod
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pickle
import warnings
from make_vector_db import ClinicalEmbeddings

file_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(file_dir)

SYSTEM_PROMPT = "You are a professional chatbot that answers questions about a " \
    "scientific paper about human breast cancer and the HER-2 oncogene. "\
    "Only answer relevant questions and politely redirect users who ask "\
    "irrelevant questions. Keep answers brief and factual."
    
RAG_SYSTEM_PROMPT = """You are a professional chatbot that answers questions about a scientific paper about human breast cancer and the HER-2 oncogene. Use the provided context to answer the user's question accurately and concisely. Keep answers brief and factual.
"""


def make_user_prompt(query):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query}
    ]
    return messages


class ChatBot(ABC):
    """ChatBot Interface (Abstract Class)
    ChatBot objects will have an answer method which takes a user query and
    generates a response.
    """
    
    @abstractmethod
    def answer(self, query):
        """Function for answering a user provided query. Given the query,
        generate chatbot response.

        Args:
            query (string): User query; intended to be a question about HER2

        Returns:
            String: LLM generated response
        """
        pass


class BasicChatBot(ChatBot):
    """
    Initialize a basic HER2 chatbot that uses an out of the box LLM as its base
    """

    def __init__(self, language_model='llama3.2:1b', system_prompt=SYSTEM_PROMPT):
        self.language_model = language_model
        self.system_prompt = system_prompt

    def _make_user_prompt(self, query):
        """Format user query into LLM prompt

        Args:
            query (string): User query

        Returns:
            List: List of 'role', 'content' dictionaries
        """
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': query}
        ]
        return messages

    def answer(self, query):
        """Function for answering a user provided query. Given the query,
        generate response from pretrained LLM.

        Args:
            query (string): User query; intended to be a question about HER2

        Returns:
            response (string): LLM generated response
        """
        messages = self._make_user_prompt(query)
        
        response = ollama.chat(model=self.language_model, messages=messages)
        return(response['message']['content'])
    
class RAGChatBot(ChatBot):
    """
    HER2 Chatbot that uses RAG with pretrained sentence transformer to provide
    context from the HER2 paper when generating responses to user queries.
    """
    
    def __init__(self, language_model='llama3.2:1b', system_prompt=RAG_SYSTEM_PROMPT,
                 top_k=3, clinical=False):
        """Initialize RAGChatBot object

        Args:
            language_model (str, optional): Which base LLM to use. Supported 
            options are 'llama3.2' and 'llama3.2:1b'. Defaults to 'llama3.2:1b'.
            
            system_prompt (str, optional): Prompt to prime LLM as chatbot. 
            Defaults to RAG_SYSTEM_PROMPT.
            
            top_k (int, optional): Number of pieces of context to consider for
            RAG. Defaults to 3.
            
            clinical (boolean, optional): Whether or not to use the Harvard Paper's clinical embeddings for RAG. Defaults to False.
        """
        self.language_model = language_model
        self.system_prompt = system_prompt
        self.top_k = top_k
        
        # preload embeddings model and vector db
        # assumes that make_vector_db has been run
        if not os.path.exists(os.path.join(project_dir, 'vecdb')):
            raise Exception('RAG chatbot expects a top level "\
                /vecdb directory containing vector db. Please run make_vector_db.py')
        
        # load correct embeddings model
        if clinical:
            collection_name = "clinical_vector_db"
            try:
                with open(os.path.join(project_dir, 'vecdb') + '/clinical_embeddings_model.pkl', 'rb') as f:
                    self.context_embeddings = pickle.load(f)
            except FileNotFoundError:
                raise Exception('RAG chatbot expects a clinical embeddings model'\
                    '../vecdb/clinical_embeddings_model.pkl to exist. Please run make_vector_db.py')
        else:
            collection_name = "RAG_vector_db"
            try:
                with open(os.path.join(project_dir, 'vecdb') + '/embeddings.pkl', 'rb') as f:
                    # surpress torch warning about loading 
                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        self.context_embeddings = pickle.load(f)
            except FileNotFoundError:
                raise Exception('RAG chatbot expects a HuggingFaceEmbeddings model"\
                    " ../vecdb/embeddings.pkl to exist. Please run make_vector_db.py')

        
        # should also check that collection name exists
        self.vectordb = Chroma(
            persist_directory=os.path.join(project_dir, 'vecdb'), 
            embedding_function=self.context_embeddings, 
            collection_name=collection_name,
            # collection_metadata={"hnsw:space": "cosine"},
            )
        # warm up vector db? To avoid Intel error on mac
        self.vectordb.similarity_search("Test", k=self.top_k)

    def _make_user_prompt(self, query, context):
        """Format user query into LLM prompt

        Args:
            query (string): User query
            context (List[string]): List of context chunks found in vector db

        Returns:
            List: List of 'role', 'content' dictionaries
        """
        
        chunks = "\n\n".join([doc.page_content for doc in context])
        
        rag_query = f"""Use the following context from the HER2 scientific paper to answer the question.
        Context: {chunks}
        
        Question: {query}

        Answer:
        """
        
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': rag_query}
        ]
        return messages

    def answer(self, query):
        """Function for answering a user provided query. Given the query,
        generate response from pretrained LLM using RAG.

        Args:
            query (string): User query; intended to be a question about HER2

        Returns:
            response (string): LLM generated response
        """        
        # get context
        
        context_chunks = self.vectordb.similarity_search(query, k=self.top_k)
        
        # format RAG prompt
        messages = self._make_user_prompt(query, context_chunks)
        
        response = ollama.chat(model=self.language_model, messages=messages)
        
        return(response['message']['content'])


def main():
    # defaulting to using a language model that can run locally on a machine
    # without using GPU. This will make it easy for someone to run and test the
    # chatbot locally on their own machine. In the future, if the chatbot were
    # to be run and served via cloud, could investigate the choice of base LLM.

    # 1b for developing; 3b for testing
    supported_llms = ['llama3.2', 'llama3.2:1b']
    parser = argparse.ArgumentParser(description="HER2 Chatbot")
    parser.add_argument("--no_rag", action="store_true", 
                        help="Disable RAG and only use out-of-the-box LLM")
    parser.add_argument("--model", type=str, default="llama3.2", 
                        help="Ollama model to use. Currently supported are 'llama3.2' and 'llama3.2:1b'")
    parser.add_argument("--clinical", action="store_true",
                        help="Use clinical embeddings for context retrieval with RAG. If --no_rag is used then this does nothing.")
    args = parser.parse_args()
    
    if args.model not in supported_llms:
        raise Exception("Currently the only supported models are 'llama3.2' and 'llama3.2:1b'")

    # initialize correct type of chatbot based on parsed commands
    if args.no_rag:
        chatbot = BasicChatBot(language_model=args.model)
    else:
        if args.clinical:
            chatbot = RAGChatBot(language_model=args.model, clinical=True)
        else:
            chatbot = RAGChatBot(language_model=args.model)
    
    print('\nWelcome to the HER2 Chatbot, which is intended to answer questions about the HER2 paper.')
    print('Type "exit" to quit.')
    # main chatbot loop
    while True:
        query = input("\nEnter question: ")
        if query.lower() == 'exit' or query.lower() == 'quit':
            break
        
        print("\nGenerating response...")
        # messages = make_user_prompt(query)
        # response = ollama.chat(model=LANGUAGE_MODEL, messages=messages)
        # print(response['message']['content'])
        
        response = chatbot.answer(query)
        print(response)


if __name__ == "__main__":
    main()
