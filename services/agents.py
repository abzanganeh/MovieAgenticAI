import os
import random
import pandas as pd
from langchain.tools import tool
from services.rag import get_vector_store
# IMPORT THE UTILS
from services.utils import parse_year_range, parse_genre, clean_mashed_names

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, 'local_data', 'processed_movies.pkl')

# Global Data Cache
df = None
if os.path.exists(CACHE_PATH):
    df = pd.read_pickle(CACHE_PATH)

# Initialize Vector Store
vectorstore = get_vector_store()

# --- AGENT TOOLS ---

@tool
def search_movies(query: str) -> str:
    """
    Finds movies based on criteria like genre, year, or rating.
    Useful for: "Find me 90s action movies", "Horror movies rated 8+".
    """
    if not vectorstore: return "Error: Database is initializing, please try again."
    
    query_lower = query.lower()
    # Using imported utils
    year_range = parse_year_range(query_lower)
    genre_filter = parse_genre(query_lower)
    
    # 1. Semantic Search
    results = vectorstore.similarity_search(query, k=30)
    
    # 2. Strict Filtering
    filtered = []
    seen_titles = set()
    
    for doc in results:
        title = doc.metadata.get('title', 'Unknown')
        if title in seen_titles: continue
        
        # Check Year
        if year_range:
            doc_year_match = parse_year_range(doc.page_content) # Re-using parser on doc content if needed, 
            # OR typically we regex the specific "Year: 1999" pattern in the text.
            # Let's keep the original logic for doc content which was specific regex:
            import re
            year_match = re.search(r'Year: (\d{4})', doc.page_content)
            if year_match:
                yr = int(year_match.group(1))
                if not (year_range[0] <= yr <= year_range[1]): continue
        
        # Check Genre
        if genre_filter and genre_filter not in doc.page_content.lower():
            continue
            
        seen_titles.add(title)
        filtered.append(doc)
        if len(filtered) >= 5: break
    
    if not filtered: return "No movies found matching those specific criteria."
    
    output = "Here is what I found:\n\n"
    for i, doc in enumerate(filtered, 1):
        lines = doc.page_content.split('\n')[:4]
        info = ' | '.join([line.split(': ')[1] for line in lines if ': ' in line])
        output += f"{i}. {info}\n"
        
    return output

@tool
def recommend_similar_movies(movie_title: str) -> str:
    """
    Suggests movies similar to a specific title.
    """
    if not vectorstore: return "Error: Database not loaded."
    
    ref_results = vectorstore.similarity_search(f"Title: {movie_title}", k=1)
    if not ref_results: return f"I couldn't find '{movie_title}'."
    
    ref_content = ref_results[0].page_content
    similar = vectorstore.similarity_search(ref_content, k=10)
    
    output = f"Movies similar to '{ref_results[0].metadata.get('title')}':\n\n"
    seen = set()
    count = 0
    
    for doc in similar:
        title = doc.metadata.get('title')
        if title == ref_results[0].metadata.get('title') or title in seen:
            continue
            
        lines = doc.page_content.split('\n')[:3]
        info = ' | '.join([line.split(': ')[1] for line in lines if ': ' in line])
        output += f"- {info}\n"
        seen.add(title)
        count += 1
        if count >= 5: break
        
    return output

@tool
def get_movie_statistics(query: str) -> str:
    """
    Calculates stats: Highest rated, averages, or counts.
    """
    if df is None: return "Error: Data cache not loaded."
    
    query_lower = query.lower()
    year_range = parse_year_range(query_lower)
    genre_filter = parse_genre(query_lower)
    
    subset = df.copy()
    if genre_filter:
        subset = subset[subset['genre'].str.contains(genre_filter, case=False, na=False)]
    if year_range:
        subset = subset[(subset['year'] >= year_range[0]) & (subset['year'] <= year_range[1])]
        
    if "average" in query_lower:
        avg = subset['rating'].mean()
        return f"Average rating: {avg:.2f}/10 (based on {len(subset)} movies)"
    
    elif any(x in query_lower for x in ["highest", "best", "top"]):
        top = subset.nlargest(5, 'rating')[['title', 'rating', 'year']]
        res = "Highest rated movies:\n"
        for _, row in top.iterrows():
            res += f"- {row['title']} ({int(row['year'])}): {row['rating']}\n"
        return res
        
    return f"Found {len(subset)} movies matching your criteria."

@tool
def generate_movie_quiz(query: str) -> str:
    """
    Generates a trivia question.
    """
    if df is None: return "Error: Data not loaded."
    
    pool = df[(df['rating'] >= 7.0) & (df['year'] >= 2000)]
    if len(pool) < 5: return "Not enough data for a quiz."
    
    movie = pool.sample(1).iloc[0]
    
    if random.choice([True, False]):
        director = movie['director']
        response = f"🎬 **Director Challenge**\n❓ Who directed **'{movie['title']}'**?"
        response += f"\n\n||INTERNAL_ANSWER_KEY: {director}||"
    else:
        cast = clean_mashed_names(movie['cast']).split(',')[0]
        response = f"🎬 **Actor Challenge**\n❓ Who starred in **'{movie['title']}'**?"
        response += f"\n\n||INTERNAL_ANSWER_KEY: {cast}||"
        
    return response

# Export tools
all_tools = [search_movies, recommend_similar_movies, get_movie_statistics, generate_movie_quiz]