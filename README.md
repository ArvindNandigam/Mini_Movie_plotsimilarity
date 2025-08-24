#  Movie Semantic Search with MongoDB + Hugging Face

This project builds a **semantic search engine** for movies using the `sample_mflix` dataset in **MongoDB Atlas** and **Hugging Face Sentence Transformers**.

---

##  Workflow

### 1. Setup & Connections
- Connects to a **MongoDB Atlas cluster** (`sample_mflix.movies` collection).
- Loads a **SentenceTransformer model** (`all-MiniLM-L6-v2`) from Hugging Face for generating embeddings.

### 2. Embedding Generation
- Removes old embeddings from the database (`plot_embedding_hf` field).
- Iterates through movie documents that contain a plot.
- Converts each movie’s **plot text into a numerical embedding vector** using the transformer model.
- Stores these embeddings back in MongoDB under the field `plot_embedding_hf`.

### 3. Query Encoding
- Takes a **search query** (e.g., `"imaginary characters from outer space at war"`).
- Converts the query into an **embedding vector** using the same model.

### 4. Vector Search in MongoDB
- Uses MongoDB Atlas’s `$vectorSearch` operator on the `plot_embedding_hf` field.
- Retrieves the **top-k most semantically similar plots** (based on cosine similarity or dot product).
- Example: If the query is about *“outer space war”*, results might include movies like **Star Wars**.

### 5. Result Display
- Prints out:
  - **Movie Title**
  - **Movie Plot Summary**
  - for the top matches.

---

##  Requirements

- Python 3.9+
- Dependencies:
  ```bash
  pip install pymongo requests numpy sentence-transformers
