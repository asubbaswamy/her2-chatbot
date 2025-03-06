from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import pickle
import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.embeddings import Embeddings
from typing import List
from tqdm import tqdm

file_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(file_dir)

def count_vocab_occurrences(text, vocabulary):
    """
    Count occurrences of vocabulary terms in text.
    
    Args:
        text (String): The text to analyze
        vocabulary (List[String]): List of words/phrases
        
    Returns:
        dict: Dictionary counting frequencies of vocabulary terms
    """
    results = {term: 0 for term in vocabulary}
    
    # Try to find longer phrases first
    sorted_vocab = sorted(vocabulary, key=len, reverse=True)
    
    remaining_text = text
    
    # Process the text for each vocabulary item
    for term in sorted_vocab:        
        is_word = len(term.split()) == 1 and term.isalnum()
        
        if is_word:
            # word boundaries
            pattern = r'\b' + re.escape(term) + r'\b'
        else:
            pattern = re.escape(term)
        
        # Find all occurrences
        matches = re.findall(pattern, remaining_text, re.IGNORECASE)
        results[term] = len(matches)
    
    return results

def aggregate_embedding(counts, embeddings, lookup):
    dim = embeddings.shape[1]
    final_embedding = np.zeros(dim)
    total = 0.
    
    # average the embeddings of the terms that appear
    for term, count in counts.items():
        if count == 0:
            continue
        index = lookup[term]
        final_embedding += count * embeddings[index]
        total += count
        
    if total == 0:
        return np.zeros(dim)
    
    return final_embedding / total



class ClinicalEmbeddings(Embeddings):
    """Custom Embedder using the Harvard Paper's Clinical Embeddings
    Weighted average of word embeddings for clinical words that appear in input text.
    """

    def __init__(self):
        self.kg = pd.read_csv(os.path.join(project_dir, 'reference_docs') + '/new_node_map_df.csv')
        self.clinical_embeddings = np.load(os.path.join(project_dir, 'reference_docs') + '/full_h_embed_hms.npy')
                
        self.vocab = self.kg.node_name.values
        self.indices = self.kg['global_graph_index'].values
        self.vocab = [v.lower() for v in self.vocab]
        self.vocab_to_index = {}
        for i in range(len(self.vocab)):
            self.vocab_to_index[self.vocab[i]] = self.indices[i]
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        result = []
        for text in tqdm(texts):
            result.append(self.embed_query(text))
        
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        q_counts = count_vocab_occurrences(text, self.vocab)
        q_embedding = aggregate_embedding(q_counts, self.clinical_embeddings, self.vocab_to_index)
        return list(q_embedding)


def main():
    print("LOADING PDF")
    loader = PyPDFLoader(os.path.join(project_dir, 'reference_docs') + '/HER2_paper_stripped.pdf')
    documents = loader.load()
    print("DONE")
    chunk_size = 500
    chunk_overlap= 50
    
    # chunking
    print("CHUNKING")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators = ["\n\n", "\n", " "]
    )
    ts_chunks = text_splitter.split_documents(documents)
    chunks = text_splitter.split_documents(documents)
    print("DONE")

    if os.path.exists(os.path.join(project_dir, 'vecdb')):
        shutil.rmtree(os.path.join(project_dir, 'vecdb'))
    
    model_name = "all-mpnet-base-v2"
    print("BUILDING VECTOR DB")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectordb = Chroma.from_documents(
        chunks,
        collection_name = "RAG_vector_db", 
        embedding=embeddings, 
        persist_directory=os.path.join(project_dir, 'vecdb')
    )
    print("DONE")
    
    # save embeddings and vec db so chatbot can open and use
    vectordb.persist()
    with open(os.path.join(project_dir, 'vecdb') + '/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    # docs = vectordb.similarity_search("By what factor did HER-2/neu amplification occur in breast cancer cases?", k=3)
    # for doc in docs:
    #     print("\nGetting relevant chunk:")
    #     print(doc.page_content)
    
    # print("Final output")
    # context = "\n\n".join([doc.page_content for doc in docs])
    # print(context)
    
    # clinical embeddings
    print("Instantiating ClinicalEmbeddings")
    embeddings = ClinicalEmbeddings()
    print("Done")
    with open(os.path.join(project_dir, 'vecdb') + '/clinical_embeddings_model.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    print("Instantiating vector db")
    vectordb = Chroma.from_documents(
        ts_chunks,
        collection_name = "clinical_vector_db", 
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=os.path.join(project_dir, 'vecdb')
    )
    print("DONE")
    
    # docs = vectordb.similarity_search("By what factor did HER-2/neu amplification occur in breast cancer cases?", k=3)
    
    # for doc in docs:
    #     print("\nGetting relevant chunk:")
    #     print(doc.page_content)
    
    # print("Final output")
    # context = "\n\n".join([doc.page_content for doc in docs])
    # print(context)
    
    
    
    
    
    

if __name__ == '__main__':
    main()