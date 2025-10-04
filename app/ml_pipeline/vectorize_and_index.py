import os
import re
import jsonlines
import chromadb
# 🚨 Fix 1: Explicitly import SentenceTransformer
from sentence_transformers import SentenceTransformer 
from chromadb.utils import embedding_functions

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = "mvp_gene_abstracts.json"
CHROMA_DB_PATH = "./chroma_db_gene_mvp"
COLLECTION_NAME = "scientific_abstract_chunks"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 5
CHUNK_OVERLAP = 1

# ==============================================================================
# EXECUTION CHECK
# ==============================================================================
print("--- SCRIPT STARTING EXECUTION ---")
print(f"Targeting input file: {INPUT_FILE}")

# ==============================================================================
# HELPER FUNCTION: CHUNKING
# ==============================================================================
def create_chunks(text, chunk_size, overlap):
    """Splits text into chunks of roughly N sentences with M overlap."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return []
    
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += (chunk_size - overlap)
        if overlap >= chunk_size:
            break
    return chunks


# ==============================================================================
# MAIN VECTORIZATION AND INDEXING PROCESS
# ==============================================================================
def build_vector_index():
    # 🚨 Debugging Check 1: Input File
    if not os.path.exists(INPUT_FILE):
        print(f"❌ FATAL ERROR: Input file not found at {INPUT_FILE}. Check file path.")
        return

    # --- 1. Load Data and Chunk ---
    print(f"\n--- 1. Loading data from {INPUT_FILE} and chunking... ---")
    data_points = []
    
    try:
        with jsonlines.open(INPUT_FILE) as reader:
            for record in reader:
                abstract = record.get('Abstract', '')
                pmid = record.get('PMID', 'NO_PMID')
                if abstract == "No Abstract Available" or len(abstract) < 50:
                    continue
                
                chunks = create_chunks(abstract, CHUNK_SIZE, CHUNK_OVERLAP)
                for i, chunk_text in enumerate(chunks):
                    data_points.append({
                        "id": f"{pmid}-{i}",
                        "text": chunk_text,
                        "metadata": {"pmid": pmid, "chunk_index": i}
                    })
    except Exception as e:
        print(f"❌ ERROR reading or parsing JSON file: {e}")
        return

    print(f"✅ Created {len(data_points)} searchable chunks.")

    # --- 2. Initialize ChromaDB ---
    print("\n--- 2. Initializing ChromaDB and Embedding Model... ---")
    
    # 🚨 Debugging Check 2: Directory Access
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print(f"✅ DB directory checked/created at {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"❌ FATAL ERROR: Directory/Permissions issue: {e}")
        return

    # 🚨 CRITICAL Model Load (Download happens here)
    try:
        print(f"🔄 Attempting to load/download embedding model: {EMBEDDING_MODEL_NAME}...")
        sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        print("✅ Embedding Model loaded successfully!")
    except Exception as e:
        print(f"❌ FATAL MODEL ERROR: Could not load or download model '{EMBEDDING_MODEL_NAME}'.")
        print(f"   Check your network connection, proxy, or run 'pip install torch' again. Error: {e}")
        return

    collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sbert_ef
    )
    print(f"✅ Collection ready: {collection.name}")

    # --- Sanity Check: Add 1 test doc ---
    # (Rest of the script continues...)
    print("\n(Sanity check) Adding 1 test doc...")
    collection.add(
        ids=["test-0"],
        documents=["This is a test sentence about gene regulation."],
        metadatas=[{"source": "unit_test"}]
    )
    print(f"Collection count after test insert: {collection.count()}")

    # --- 3. Vectorize and Index ---
    print(f"\n--- 3. Indexing {len(data_points)} chunks into ChromaDB... ---")
    ids = [d['id'] for d in data_points]
    documents = [d['text'] for d in data_points]
    metadatas = [d['metadata'] for d in data_points]

    BATCH_INSERT_SIZE = 1000
    for i in range(0, len(data_points), BATCH_INSERT_SIZE):
        batch_ids = ids[i:i + BATCH_INSERT_SIZE]
        batch_documents = documents[i:i + BATCH_INSERT_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_INSERT_SIZE]

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        print(f"✅ Indexed batch {i//BATCH_INSERT_SIZE + 1} "
              f"({len(batch_ids)} items)")

    print(f"\n✅ Vectorization complete. Total items in collection: {collection.count()}")

    # --- 4. Verification Query ---
    print("\n--- 4. Testing Semantic Search ---")
    test_query = "What novel compound inhibits the WNT signaling pathway?"
    results = collection.query(
        query_texts=[test_query],
        n_results=3,
        include=['documents', 'distances', 'metadatas']
    )

    print(f"Query: '{test_query}'")
    for i, doc in enumerate(results['documents'][0]):
        pmid = results['metadatas'][0][i].get('pmid', 'NA')
        dist = results['distances'][0][i]
        print(f"\nResult {i+1} (Distance: {dist:.4f} | PMID: {pmid}):")
        print(f"Snippet: {doc[:150]}...")


# ==============================================================================
if __name__ == "__main__":
    # The crash must now be either an import failure or a model load failure.
    build_vector_index()