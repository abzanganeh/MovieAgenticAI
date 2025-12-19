import re
from rapidfuzz import fuzz

def parse_year_range(query_lower):
    """
    Extracts year range (e.g., '90s' -> 1990-1999) from text.
    """
    # Handle decades ("90s", "1990s")
    decade_match = re.search(r'\b(\d{2,4})\s?\'?s\b', query_lower)
    if decade_match:
        digits = decade_match.group(1)
        # Convert "90" to "1990"
        start_year = int("19" + digits) if len(digits) == 2 else int(digits)
        return (start_year, start_year + 9)

    # Handle specific years ("2023")
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query_lower)
    if year_match:
        specific_year = int(year_match.group(1))
        return (specific_year, specific_year)
    return None

def parse_genre(query_lower):
    """
    Finds the closest matching genre in the text using Fuzzy Logic.
    """
    genres = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime',
              'documentary', 'drama', 'fantasy', 'family', 'history', 'horror',
              'musical', 'mystery', 'romance', 'sci-fi', 'reality-tv']
    
    # Common alias
    if 'scifi' in query_lower: return 'sci-fi'

    # 1. Exact Match
    for genre in genres:
        if f" {genre} " in f" {query_lower} ": return genre

    # 2. Fuzzy Match (Typo tolerance)
    for word in query_lower.split():
        for genre in genres:
            if fuzz.ratio(genre, word) > 80:
                return genre
    return None

def clean_mashed_names(text):
    """Fixes 'LeonardoDiCaprio' -> 'Leonardo, DiCaprio'"""
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ', ', str(text))