import pymongo
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

client = pymongo.MongoClient(
    "mongodb+srv://<user>:<pass>@cluster0.lrotplk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client.sample_mflix
collection = db.movies

hf_token = "<token>"
embedding_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {hf_token}"}

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(text: str) -> list[float]:
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()

print("Clearing old embeddings...")
collection.update_many({}, {"$unset": {"plot_embedding_hf": ""}})

print("Generating and storing embeddings...")
for doc in collection.find({"plot": {"$exists": True}}).limit(50):
    vec = generate_embedding(doc["plot"])
    collection.update_one({"_id": doc["_id"]}, {"$set": {"plot_embedding_hf": vec}})

print("Embeddings stored successfully!")

query = "imaginary characters from outer space at war"
query_vec = generate_embedding(query)

print(f"\nRunning semantic search for: '{query}'\n")

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_vec,
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch"
        }
    }
])

for document in results:
    print(f'Movie Name: {document['title']}\nMovie Plot: {document['plot']}\n')
