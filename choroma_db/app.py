import chromadb
chroma_client = chromadb.Client()


collection = chroma_client.create_collection(name="my_collection")

documents = [
    {"id": "1", "text": "This is the first document."},
    {"id": "2", "text": "This is the second document."},
    {"id": "3", "text": "This is the third document."}]


collection.add(ids=[doc["id"] for doc in documents], documents=[doc["text"] for doc in documents])
# for doc in documents:
#     collection.upsert(ids=doc["id"], documents=[doc["text"]])

result = collection.query(query_texts=["first document"], n_results=1)

# print(chroma_client.list_collections())
print(result)



#simillarity search with scores 

