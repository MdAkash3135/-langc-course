#pg vector create index using hnsw

# create index on documets using hnsw(embedding_vector_cosine_ops) with (m=16, ef_construction=200, ef_search=50, max_elements=1000000);


from http import client


collection = client.create_collection(
    name="scallign_rag",
    metadata= {
  'hnsw:m': 16,
  'hnsw:ef_construction': 200,
  'hnsw:ef_search': 50,
  'hnsw:max_elements': 1000000
    }
)