import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'local_data', 'movies.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'plots')

def run_eda():
    print(f"--- Starting EDA ---")
    print(f"Reading data from {DATA_PATH}...")
    
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 1. Clean Column Names (Same as Ingest)
    # We rename them first so the plots have nice labels
    column_mapping = {
        'Title': 'title', 'Year': 'year', 'Genre': 'genre',
        'Director': 'director', 'Description': 'overview',
        'Certificates': 'certificates', 'MetaScore': 'metascore',
        'IMDb Rating': 'rating', 'Star Cast': 'cast',
        'Poster-src': 'poster', 'Duration (minutes)': 'duration'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Fill numeric nulls for plotting
    if 'year' in df.columns: df['year'] = df['year'].fillna(df['year'].median())
    if 'rating' in df.columns: df['rating'] = df['rating'].fillna(df['rating'].mean())

    print(f"Generating charts based on {len(df)} movies...")

    # Set the style
    sns.set_theme(style="whitegrid")
    
    # --- PLOT 1: Genre Distribution ---
    plt.figure(figsize=(10, 6))
    # Get top 15 genres
    genre_counts = df['genre'].value_counts().head(15)
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="husl")
    plt.title('Top 15 Genres', fontsize=16)
    plt.xlabel('Number of Movies')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'genre_distribution.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()

    # --- PLOT 2: Rating Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating'], bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(df['rating'].mean(), color='red', linestyle='--', label=f'Mean: {df["rating"].mean():.1f}')
    plt.title('Distribution of IMDb Ratings', fontsize=16)
    plt.xlabel('Rating')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'rating_distribution.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()

    # --- PLOT 3: Movies per Year ---
    plt.figure(figsize=(12, 6))
    # Filter reasonable years (e.g. 1900-2025)
    years = df['year']
    years = years[(years >= 1900) & (years <= 2025)]
    year_counts = years.value_counts().sort_index()
    
    plt.plot(year_counts.index, year_counts.values, color='darkblue', linewidth=2)
    plt.fill_between(year_counts.index, year_counts.values, color='darkblue', alpha=0.1)
    plt.title('Movies Released Per Year', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'movies_per_year.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()

    # --- PLOT 4: Rating vs Year (Quality Trend) ---
    plt.figure(figsize=(12, 6))
    plot_data = df[['year', 'rating']].dropna()
    plot_data = plot_data[(plot_data['year'] >= 1900) & (plot_data['year'] <= 2025)]
    
    plt.scatter(plot_data['year'], plot_data['rating'], alpha=0.3, s=20, color='purple')
    
    # Trend line
    z = np.polyfit(plot_data['year'], plot_data['rating'], 1)
    p = np.poly1d(z)
    plt.plot(plot_data['year'], p(plot_data['year']), "r--", linewidth=2, label='Trend')
    
    plt.title('Rating Trends Over Time', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('IMDb Rating')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'rating_vs_year.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()

    print("--- EDA Complete ---")

if __name__ == "__main__":
    run_eda()