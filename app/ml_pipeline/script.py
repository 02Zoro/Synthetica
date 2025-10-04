import pandas as pd
from Bio import Entrez
import time
import json
import ssl
import re

# ==============================================================================
# 1. CONFIGURATION: UPDATE THESE FIELDS
# ==============================================================================
# 🚨 IMPORTANT: Replace these placeholders with your actual details
Entrez.email = "stutisingh876@gmail.com" 
Entrez.api_key = "82d57cc1f04bd185f1b0739a00e73c630508" 

# Search Parameters (Using the successfully broadened query)
SEARCH_TERM = "(gene OR protein) AND (disease model OR signaling pathway)" 
DB = "pubmed"
RETMAX = 5000  # Target number of abstracts for the MVP
BATCH_SIZE = 500 # Number of records to fetch in a single API call
OUTPUT_FILE = "mvp_gene_abstracts.json"
# ==============================================================================

def search_pubmed(query, db, retmax):
    """Searches PubMed and returns a list of PMIDs (IDs) and WebEnv/QueryKey."""
    print(f"--- 1. Searching {db} for: '{query}' (Max {retmax} records) ---")
    
    handle = Entrez.esearch(
        db=db, 
        term=query, 
        retmax=retmax, 
        usehistory="y", 
        retmode="xml"
    )
    record = Entrez.read(handle)
    handle.close()
    
    count = int(record['Count'])
    
    if count == 0:
        return [], None, None
        
    web_env = record.get('WebEnv')
    query_key = record.get('QueryKey')
    
    if count <= retmax:
        print(f"Found {count} articles. Fetching all via ID list.")
        return record['IdList'], None, None
    else:
        print(f"Found {count} articles. Fetching the top {retmax} using history.")
        return record['IdList'], web_env, query_key


def fetch_records(id_list, web_env=None, query_key=None, start_offset=0):
    """Fetches full records in batches using Entrez.read for robust XML parsing."""
    papers = []
    num_ids = len(id_list)
    
    for start in range(start_offset, num_ids, BATCH_SIZE):
        end = min(num_ids, start + BATCH_SIZE)
        
        # Determine the parameters to fetch
        if web_env and query_key:
            fetch_params = {
                'retmax': BATCH_SIZE,
                'retstart': start,
                'query_key': query_key,
                'WebEnv': web_env
            }
            fetch_ids = None 
        else:
            fetch_params = {}
            fetch_ids = id_list[start:end]
        
        print(f"Fetching records from {start+1} to {end}...")

        try:
            fetch_handle = Entrez.efetch(
                db=DB, 
                id=fetch_ids, 
                retmode="xml", 
                **fetch_params
            )
            
            # 💡 CRITICAL FIX 1: Use Entrez.read() for reliable XML parsing of the batch
            records = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            # 💡 CRITICAL FIX 2: Iterate over the correct list structure: 'PubmedArticle'
            for record in records.get('PubmedArticle', []):
                pmid = "Unknown"  # Initialize safely for error handling
                try:
                    # Safely extract PMID
                    pmid = str(record.get('MedlineCitation', {}).get('PMID', 'Unknown'))
                    
                    # Extract article and abstract content
                    article = record.get('MedlineCitation', {}).get('Article', {})
                    title = article.get('ArticleTitle', 'No Title')
                    abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
                    
                    if isinstance(abstract_parts, list) and abstract_parts:
                        # Join all abstract parts (e.g., Background, Methods, Results)
                        abstract = " ".join([str(text) for text in abstract_parts]).strip()
                    else:
                        abstract = "No Abstract Available"
                        
                    papers.append({
                        'PMID': pmid, 
                        'Title': str(title).strip(), 
                        'Abstract': abstract,
                        'Source': 'PubMed'
                    })
                
                except Exception as e:
                    # Log error but continue to the next record
                    print(f"Warning: Failed to parse content for PMID: {pmid}. Error: {e}")

            time.sleep(0.5) 

        except Exception as e:
            # Handle API/Network errors
            print(f"FATAL API ERROR during batch fetch: {e}. Waiting 10s and skipping batch.")
            time.sleep(10) 

    return papers

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Search for IDs ---
    id_list, web_env, query_key = search_pubmed(SEARCH_TERM, DB, RETMAX)

    if not id_list:
        print("No articles found. Exiting.")
    else:
        print(f"--- 2. Fetching and Parsing {len(id_list)} Abstracts ---")
        
        # --- 3. Fetch Records ---
        all_papers = fetch_records(id_list, web_env, query_key)

        # --- 4. Save to JSON ---
        if all_papers:
            df = pd.DataFrame(all_papers)
            df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)
            print(f"\n✅ SUCCESS! Saved {len(df)} abstracts to {OUTPUT_FILE}")
        else:
            print("\n❌ FAILED to retrieve any abstracts after search.")