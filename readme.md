# RAG Weather Agent (Quick Start)


[![Watch the video](https://cdn.loom.com/sessions/thumbnails/78b956d2260941ddae5a01c324773a1e-3d715dee3f6bcb46-full-play.gif)](https://www.loom.com/share/YOUR_VIDEO_ID)


A Python agent that answers weather questions and queries PDF documents using Retrieval-Augmented Generation (RAG).

## Features
- Real-time weather lookup (OpenWeatherMap)
- RAG over PDFs (Qdrant vector store)
- LangGraph agentic routing
- Modular, testable code

## Setup
1. **Clone & Install**
   ```bash
   git clone https://github.com/<your-username>/rag-weather-agent.git
   cd rag-weather-agent
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. **Configure**
   - Copy `.env.template` to `.env` and add your API keys (OpenAI, OpenWeatherMap, Qdrant).

## Usage
- **Ingest PDFs:**
  ```bash
  python scripts/ingest_data.py
  ```
- **Run Agent (CLI):**
  ```bash
  python main.py
  ```
- **Run Tests:**
  ```bash
  python -m unittest discover -s tests
  ```

## Project Structure
- `agents/` – LangGraph agent
- `tools/` – Weather & retriever tools, prompts
- `data/` – PDF files
- `scripts/` – Ingestion scripts
- `main.py` – CLI entry point

## License
MIT © 2025
