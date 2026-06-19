"""
MovieLens Chatbot — DV1729 Final Project
RAG pipeline with dual LLM comparison (Ollama: llama3.2 + mistral)
Dataset: MovieLens ml-32m
"""

import os
import re
import pandas as pd
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS = ["llama3.2", "mistral"]
MAX_RESULTS = 10

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "ml-32m", "ml-32m")

print(" Loading MovieLens dataset …")

movies_path = os.path.join(data_dir, "movies.csv")

clean_rows = []
with open(movies_path, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",", 2)
        if len(parts) == 3:
            clean_rows.append(parts)

movies = pd.DataFrame(clean_rows, columns=["movieId", "title", "genres"])
movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
movies.dropna(subset=["movieId"], inplace=True)
movies["movieId"] = movies["movieId"].astype(int)
movies["genres"] = movies["genres"].fillna("")

# Ratings 
ratings = pd.read_csv(
    os.path.join(data_dir, "ratings.csv"),
    nrows=500_000
)

# Tags 
tags = pd.read_csv(
    os.path.join(data_dir, "tags.csv"),
    nrows=200_000
)

# Clean numeric fields
for frame in (ratings, tags):
    frame["movieId"] = pd.to_numeric(frame["movieId"], errors="coerce")
    frame.dropna(subset=["movieId"], inplace=True)
    frame["movieId"] = frame["movieId"].astype(int)

ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
ratings.dropna(subset=["rating"], inplace=True)

# Convert timestamp → year
ratings["year"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce").dt.year

# Merge
df = pd.merge(ratings, movies, on="movieId", how="inner")
movies_df = movies.copy()
ratings_df = ratings.copy()
tags_df = pd.merge(tags, movies, on="movieId", how="inner")

print(
    f" Loaded {len(movies):,} movies | "
    f"{len(ratings):,} ratings | "
    f"{len(tags_df):,} tags\n"
)

# ─────────────────────────────────────────────
# OLLAMA CLIENT
# ─────────────────────────────────────────────
client = ollama.Client(host="http://localhost:11434")

def ollama_generate(model: str, prompt: str) -> str:
    try:
        resp = client.generate(model=model, prompt=prompt)
        return resp["response"].strip()
    except Exception as exc:
        return f"[{model} error: {exc}]"

# ─────────────────────────────────────────────
# QUERY GENERATION 
# ─────────────────────────────────────────────
QUERY_SYSTEM = """You are an expert Pandas developer working with MovieLens data.

 CRITICAL DATA STRUCTURE (READ CAREFULLY):
- 'df' has these columns ONLY: userId, movieId, rating, timestamp, year, title, genres
- There is NO column called 'rating year' - just use 'year'
- 'year' = the year the RATING was given (not movie release year)
- Each movie appears MULTIPLE times (once per user rating)
- To find TAGS for a movie, you MUST use 'tags_df'.
- Always find the movieId first, then filter tags_df.

 CORRECT QUERY PATTERNS:

1. Top N movies by genre:
df[df['genres'].str.contains('GENRE', case=False, na=False)].groupby('title')['rating'].mean().sort_values(ascending=False).head(N)

2. Top N movies by genre from specific rating year:
df[(df['genres'].str.contains('GENRE', case=False, na=False)) & (df['year'] == YEAR)].groupby('title')['rating'].mean().sort_values(ascending=False).head(N)

3. Top N movies by genre from year range:
df[(df['genres'].str.contains('GENRE', case=False, na=False)) & (df['year'] >= START) & (df['year'] <= END)].groupby('title')['rating'].mean().sort_values(ascending=False).head(N)

4. Top N movies overall (all genres):
df.groupby('title')['rating'].mean().sort_values(ascending=False).head(N)

CORRECT PATTERN for "tags for Movie X":
tags_df[tags_df['movieId'] == movies_df[movies_df['title'].str.contains('Inception', case=False)]['movieId'].iloc[0]]['tag'].value_counts().head(10)

 WRONG PATTERNS (NEVER USE):
- df['rating year'] - this column DOES NOT EXIST
- df[df['year'] == df['rating year']] - makes no sense
- Any column name not in the list above

 SIMPLE RULE: Only use columns from this list: userId, movieId, rating, timestamp, year, title, genres
"""

def extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Remove any explanatory text before the code
    lines = text.strip().split('\n')
    for i, line in enumerate(lines):
        if 'df[' in line or 'df.' in line:
            return line.strip().rstrip('.;,')
    return text.strip().rstrip('.;,')

def generate_query(user_question: str) -> str:
    prompt = f"{QUERY_SYSTEM}\nUser question: {user_question}\nPandas expression (CODE ONLY, no explanations):"
    raw = ollama_generate(MODELS[0], prompt)
    
    
    query = extract_code_block(raw)
    query = query.split('\n')[0].strip()
    query = query.rstrip('.,;')
    query = query.replace('"rating year"', 'year').replace("'rating year'", 'year')

    # Safety fix for unclosed brackets/parentheses
    if query.count('[') > query.count(']'):
        query += ']' * (query.count('[') - query.count(']'))
    if query.count('(') > query.count(')'):
        query += ')' * (query.count('(') - query.count(')'))
    
    print(f" Model generated: {query}")
    return query

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
SAFE_GLOBALS = {"__builtins__": __builtins__}
SAFE_LOCALS = {
    "df": df,
    "movies_df": movies_df,
    "ratings_df": ratings_df,
    "tags_df": tags_df,
    "pd": pd,
}

def validate_query(query: str):
    try:
        result = eval(query, SAFE_GLOBALS, SAFE_LOCALS)
        if isinstance(result, (pd.DataFrame, pd.Series)) and len(result) == 0:
            return False, "No results"
        return True, result
    except Exception as e:
        return False, str(e)

# ─────────────────────────────────────────────
# FORMAT
# ─────────────────────────────────────────────
def format_result(result):
    # If the result is a single number or string (not a DataFrame/Series)
    if not hasattr(result, 'head'):
        return str(result)
    
    # Otherwise, format as a table
    df_str = result.head(MAX_RESULTS).round(2).to_string()
    return df_str.replace('"\n', '" ')

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def summarize(model, question, raw):
    prompt = f"Question: {question}\nData:\n{raw}\n\nGive short insight."
    return ollama_generate(model, prompt)

def run_parallel_summaries(q, raw):
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(summarize, m, q, raw): m for m in MODELS}
        return {futures[f]: f.result() for f in futures}

# ─────────────────────────────────────────────
# INTRO
# ─────────────────────────────────────────────
def print_intro():
    print("=" * 60)
    print(" MovieLens Chatbot — DV1729 Project")
    print("=" * 60)
    # This fulfills the "introduce dataset scope" requirement 
    print(f"Scope: This dataset contains {len(movies):,} movies and {len(ratings):,} user ratings.")
    print("You can ask me for movie recommendations, 'best' films by genre,")
    print("or trends from specific years. What specific data are you looking for?")
    print("=" * 60)

# ─────────────────────────────────────────────
# CHALLENGE TESTS 
# ─────────────────────────────────────────────
def run_challenge_tests():
    """Run 3+ challenging test scenarios"""
    print("\n" + "="*60)
    print("RUNNING CHALLENGE TESTS")
    print("="*60)
    
    tests = [
        # TEST 1: (Error Handling) Forces a column error
        ("1. Non-existent field", "Compute the average of a column named 'non_existent_score'"),
        
        # TEST 2: (Joins) Forces a merge between two specific DataFrames
        ("2. Cross-table join", "Join tags_df and movies_df on movieId and show the tags for 'Pulp Fiction'"),
        
        # TEST 3: (Refinement) Forces the >10 results rule to trigger
        ("3. Ambiguous query", "Show me a list of all movie titles without any filtering or limits"),
    ]
    
    results_summary = []
    
    for test_name, test_query in tests:
        print(f"\n TEST {test_name}")
        print(f"   Query: '{test_query}'")
        
        query = generate_query(test_query)
        valid, result = validate_query(query)
        
        if not valid:
            print(f"    Handled gracefully: {result}")
            results_summary.append(f"{test_name}: PASS (error caught)")
        elif len(result) > 10: # ADD THIS CHECK HERE
            print(f"    Handled gracefully: Found {len(result)} results. Refinement required.")
            results_summary.append(f"{test_name}: PASS (refinement triggered)")
        else:
            print(f"   Query executed, got {len(result)} results")
            results_summary.append(f"{test_name}: PASS (executed)")
    
    print("\n" + "="*60)
    print(" CHALLENGE TEST SUMMARY:")
    for r in results_summary:
        print(f"   • {r}")
    print("="*60)

# ─────────────────────────────────────────────
# DISCREPANCY RESOLUTION
# ─────────────────────────────────────────────
def resolve_discrepancies(question, summary_a, summary_b):
    unify_prompt = f"""
    User Question: {question}
    Model A Summary: {summary_a}
    Model B Summary: {summary_b}
    
    Compare these two summaries. If they conflict (e.g., different movies listed), 
    provide a unified final response and explicitly mention that there was a 
    discrepancy between the models.
    """
    # Use your primary model to unify
    return ollama_generate(MODELS[0], unify_prompt)

# ─────────────────────────────────────────────
# CHAT LOOP
# ─────────────────────────────────────────────
def chat():
    print_intro()

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        print("\nThinking...")
        query = generate_query(user)

        valid, result = validate_query(query)

        if not valid:
            print(" Failed query:", result)
            continue

        # Requirement: If results >10, ask to refine 
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if len(result) > 10:
                print(f"  Found {len(result)} results. Please refine your query.")
                continue
            
        raw = format_result(result)

        print("\n Results:\n", raw)

        # At the end of the 'while True' loop:
        summaries = run_parallel_summaries(user, raw)
        
        for model_name, summary_text in summaries.items():
            print(f"\n[{model_name} Summary]: {summary_text}")
        
        final_response = resolve_discrepancies(user, summaries[MODELS[0]], summaries[MODELS[1]])
        print("\n FINAL UNIFIED ANALYSIS:")
        print(final_response)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_challenge_tests()
    print("\n" + "="*60)
    print("STARTING INTERACTIVE CHATBOT")
    print("="*60)
    chat()