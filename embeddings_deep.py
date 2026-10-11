from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import numpy as np
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def basic_embeddings():
    text = "what is machine learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"vector dimensions: {len(single_embedding)}")
    print(f"1st 5 dimensions: {single_embedding[:5]}")
    print(f"vector norm: {np.linalg.norm(single_embedding):.4f}")

# basic_embeddings()

def batch_embeddings():
    texts = [
        "what is machine learning?",
        "what is deep learning?",
        "what is reinforcement learning?",
    ]
    batch_embeddings = embeddings.embed_documents(texts)
    print(f"batch vector dimensions: {len(batch_embeddings)}")
    print(f"1st 5 dimensions of 1st vector: {batch_embeddings[0][:5]}")
    print(f"vector norm of 1st vector: {np.linalg.norm(batch_embeddings[0]):.4f}")

def similarity_search():

    # Documents

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # embed documents and query
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    # compute cosine similarities
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")
