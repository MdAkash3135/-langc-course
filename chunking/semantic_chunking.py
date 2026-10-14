from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker

from langchain_core.documents import Document
import os
from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


document = """# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""

document = [Document(page_content=document) ]

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
)

recursive_chunks = recursive_splitter.split_text(document)


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""],
)


recursive_chunk = recursive_splitter.split_documents([document])

# print("Recursive Character Text Splitter:")
# for i, chunk in enumerate(recursive_chunk):
#     print(f"Chunk {i + 1}: {chunk.page_content[:100]}...")  # Print first 100 characters of each chunk



semantic_splitter = TextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""],
    embeddings=embeddings,
)

semantic_chunker = SemanticChunker(
    text_splitter=semantic_splitter,
    embeddings=embeddings,
    similarity_threshold=0.8,
)