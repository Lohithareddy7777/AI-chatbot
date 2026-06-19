# 🎬 MovieLens Chatbot – DV1729 Final Project

### Author: Lohitha Naga Gayathri Yarram
### Blekinge Institute of Technology (BTH)

---

## Project Overview

MovieLens Chatbot is a natural language movie analytics system built using Python, Pandas, and locally hosted Large Language Models (LLMs) through Ollama.

The chatbot allows users to ask movie-related questions in plain English. It automatically converts these questions into executable Pandas queries, retrieves information from the MovieLens dataset, and generates concise insights using multiple AI models.

The project demonstrates how LLMs can be integrated with structured datasets to support data exploration, movie analysis, and recommendation-style queries.

---

## Objectives

The objectives of this project are to:

- Convert natural language questions into Pandas queries.
- Retrieve and analyze information from the MovieLens dataset.
- Compare outputs from multiple language models.
- Handle invalid, ambiguous, and complex user queries.
- Generate concise insights from structured movie data.
- Demonstrate robust chatbot behavior through challenge testing.

---

## Features

### Natural Language Querying

Users can ask questions such as:

- What are the top-rated Action movies?
- Show the best Comedy movies from 2020.
- What are the most common tags for Inception?
- Which movies have the highest average ratings?
- Recommend highly rated Sci-Fi movies.

### Automated Query Generation

The chatbot uses **Llama 3.2** to translate natural language questions into executable Pandas expressions.

### Query Validation

Generated queries are validated before execution to ensure:

- Correct syntax
- Valid dataset fields
- Meaningful results

### Multi-Model Analysis

Results are analyzed independently by:

- Llama 3.2
- Mistral

### Discrepancy Resolution

When the two models produce different interpretations, the chatbot generates a unified final response.

### Robust Error Handling

The chatbot handles:

- Invalid queries
- Missing columns
- Empty results
- Large result sets
- Ollama connection failures

### Challenge Testing

The system automatically executes challenge tests to evaluate robustness before entering interactive mode.

---

## Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Core implementation |
| Pandas | Data processing and querying |
| Ollama | Local LLM hosting |
| Llama 3.2 | Query generation and final response synthesis |
| Mistral | Independent result summarization |
| ThreadPoolExecutor | Parallel model execution |
| MovieLens Dataset | Movie ratings and metadata |

---

## Dataset

This project uses the **MovieLens ml-32m dataset**.

Files used:

- `movies.csv`
- `ratings.csv`
- `tags.csv`

The chatbot loads:

- Movie metadata
- User ratings
- User-generated tags

For performance reasons, the implementation loads:

- 500,000 ratings
- 200,000 tags

along with the complete movie metadata dataset.

---

## System Architecture

```text
User Question
      │
      ▼
Llama 3.2
(Query Generation)
      │
      ▼
Pandas Query
      │
      ▼
Validation
      │
      ▼
MovieLens Dataset
      │
      ▼
Query Results
      │
      ▼
Parallel Summaries
 ┌─────────────┐
 │ Llama 3.2  │
 └─────────────┘
 ┌─────────────┐
 │  Mistral   │
 └─────────────┘
      │
      ▼
Discrepancy Resolution
      │
      ▼
Final Unified Analysis
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install pandas ollama
```

---

## Installing Ollama Models

Download and install Ollama:

https://ollama.com/download

Start Ollama:

```bash
ollama serve
```

Download the required models:

```bash
ollama pull llama3.2
ollama pull mistral
```

Verify installation:

```bash
ollama list
```

---

## Project Structure

```text
MovieLens-Chatbot/
│
├── a.py
│
├── ml-32m/
│   └── ml-32m/
│       ├── movies.csv
│       ├── ratings.csv
│       └── tags.csv
│
├── README.md
└── requirements.txt
```

---

## Running the Chatbot

Ensure Ollama is running:

```bash
ollama serve
```

Run the application:

```bash
python a.py
```

The chatbot will:

1. Load the MovieLens dataset.
2. Run challenge tests.
3. Start the interactive chat session.

---

## Example Queries

### Top Rated Movies

```text
Show the highest-rated Action movies.
```

### Genre-Based Analysis

```text
What are the best Sci-Fi movies?
```

### Year-Based Analysis

```text
Show the top Comedy movies from 2020.
```

### Tag Analysis

```text
Show the most common tags for Inception.
```

### Popular Movies

```text
Which movies have received the most ratings?
```

---

## Challenge Tests

The chatbot automatically executes three challenge scenarios.

### Test 1 – Invalid Field Query

Purpose:

- Verify graceful handling of invalid dataset fields.

Example:

```text
Compute the average of a column named non_existent_score.
```

### Test 2 – Cross-Table Join

Purpose:

- Verify joins between movie and tag data.

Example:

```text
Join tags_df and movies_df on movieId and show tags for Pulp Fiction.
```

### Test 3 – Ambiguous Query

Purpose:

- Verify refinement requests for overly broad queries.

Example:

```text
Show all movie titles.
```

---

## Error Handling

The chatbot is designed to handle:

### Invalid Queries

```text
Failed query: invalid syntax
```

### Missing Results

```text
No results
```

### Ollama Connection Errors

```text
Failed to connect to Ollama
```

### Large Result Sets

When more than 10 results are returned, the chatbot asks the user to refine the query.

---

## Limitations

- Requires Ollama to be installed and running locally.
- Depends on Llama 3.2 and Mistral being available.
- Uses a subset of ratings and tags for performance.
- Query quality depends on LLM-generated Pandas expressions.

---

## Future Improvements

Potential enhancements include:

- Web-based user interface
- Advanced recommendation algorithms
- Better query generation prompts
- Additional movie datasets
- User preference tracking
- Visualization dashboards
- Expanded analytics features

---

## Academic Context

This project was developed as part of the **DV1729 course** at **Blekinge Institute of Technology (BTH)**.

The project demonstrates the integration of Large Language Models with structured datasets for conversational data analysis and movie exploration.

---

## Author

**Lohitha Naga Gayathri Yarram**

DV1729 Final Project

Blekinge Institute of Technology (BTH)
