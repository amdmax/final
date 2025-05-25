# Interactive Learning System with LangChain

This project is a Python-based interactive learning platform that allows students to:
- Manage learning topics (create, delete, list)
- Track progress within each topic
- Interact with the system, answer questions, and progress their knowledge

## Features
- Topic management (CRUD)
- Student progress tracking
- Interactive Q&A using LangChain

## Setup
1. Create and activate a virtual environment:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install langchain
   ```

## Environment Variables

Copy `.env.example` to `.env` and fill in your OpenAI and Perplexity API keys:

```sh
cp .env.example .env
# Edit .env and set your keys
```

You can set these variables in your shell or use a tool like `python-dotenv` to load them automatically.

- `OPENAI_API_KEY` — required for LLM features
- `PERPLEXITY_API_KEY` — required for Perplexity features (if used)

## Usage
Run the main script to start the interactive session:
```sh
python main.py
```

## Requirements
- Python 3.8+
- LangChain

---
