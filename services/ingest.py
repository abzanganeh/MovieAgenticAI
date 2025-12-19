import os
import pandas as pd
from langchain_core.documents import Document
# We import the function, but we won't run it if the DB exists
from services.rag import create_vector_db

# Define paths (Using the new local_data folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'local_data', 'movies.csv')
CACHE_PATH = os.path.join(BASE_DIR, 'local_data', 'processed_movies.pkl')
INDEX_PATH = os.path.join(BASE_DIR, 'local_data', 'faiss_index')

def make_description(row):
    """Combines columns into a rich text description."""
    try: year = int(row['year'])
    except: year = "Unknown"
        
    try: duration = int(row['duration'])
    except: duration = "Unknown"

    desc = (
        f"Title: {row['title']}\n"
        f"Year: {year}\n"
        f"Genre: {row['genre']}\n"
        f"Rating: {row['rating']}/10\n"
        f"Director: {row['director']}\n"
        f"Cast: {row['cast']}\n"
        f"Duration: {duration} minutes\n"
        f"Certificate: {row['certificates']}"
    )

    if pd.notna(row['metascore']) and row['metascore'] != 'Unknown':
        desc += f"\nMetaScore: {row['metascore']}/100"

    return desc

def ingest_movies():
    print("--- Starting Data Ingestion ---")
    df = None

    # --- PHASE 1: DATA LOADING (The Cache Logic) ---
    if os.path.exists(CACHE_PATH):
        print(f"✅ Found cached data at {CACHE_PATH}")
        print("   Skipping CSV processing.")
        df = pd.read_pickle(CACHE_PATH)
    else:
        # If cache DOESN'T exist, we do the hard work
        print(f"Reading raw CSV from {DATA_PATH}...")
        try:
            df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
            
            # 1. Rename Columns (Standardize)
            column_mapping = {
                'Title': 'title', 'Year': 'year', 'Genre': 'genre',
                'Director': 'director', 'Description': 'overview',
                'Certificates': 'certificates', 'MetaScore': 'metascore',
                'IMDb Rating': 'rating', 'Star Cast': 'cast',
                'Poster-src': 'poster', 'Duration (minutes)': 'duration'
            }
            df.rename(columns=column_mapping, inplace=True)

            # 2. Clean Data (Impute)
            text_cols = ['director', 'cast', 'genre', 'certificates', 'overview']
            for col in text_cols:
                if col in df.columns: df[col] = df[col].fillna('Unknown')
            
            if 'year' in df.columns: df['year'] = df['year'].fillna(df['year'].median())
            if 'duration' in df.columns: df['duration'] = df['duration'].fillna(df['duration'].median())

            # 3. Create Descriptions
            print("Generating descriptions...")
            df['description'] = df.apply(make_description, axis=1)
            
            # 4. Save Cache
            print(f"💾 Saving cleaned dataframe to {CACHE_PATH}")
            df.to_pickle(CACHE_PATH)

        except Exception as e:
            print(f"❌ Error during data processing: {e}")
            return

    # --- PHASE 2: HANDOFF TO RAG ---
    
    # Check if the Vector Database already exists
    if os.path.exists(INDEX_PATH):
        print(f"✅ Vector Database already exists at '{INDEX_PATH}'.")
        print("   Skipping vector creation.")
        return

    # If we are here, we need to build the DB
    print("⚡ Converting rows to LangChain Documents...")
    documents = []
    for _, row in df.iterrows():
        # We attach metadata 'title' here. 
        # This is CRITICAL for the Agents later.
        doc = Document(
            page_content=row['description'],
            metadata={'title': row['title']}
        )
        documents.append(doc)

    print(f"⚡ Sending {len(documents)} documents to RAG service for chunking & indexing...")
    
    # We hand off the raw documents. RAG will decide how to chunk them.
    create_vector_db(documents)
    print("✅ Success! Ingestion Complete.")

if __name__ == "__main__":
    ingest_movies()