# RAG Weather Agent

A LangGraph-based AI agent that combines **real-time weather lookups** with **Retrieval-Augmented Generation (RAG)** over PDF documents. The agent autonomously decides whether to call the OpenWeatherMap API or query a Qdrant vector store, then uses an LLM to synthesize the final answer.

---

## Features

| Capability | Description |
|------------|-------------|
| **Weather Tool** | Fetches live weather data via OpenWeatherMap API |
| **RAG Retriever** | Retrieves and grades document chunks from Qdrant; rewrites queries when relevance is low |
| **Agentic Routing** | LangGraph conditional edges route queries to the appropriate tool |
| **LangSmith Tracing** | Full observability of LLM calls, tool invocations, and latencies |
| **Modular Prompts** | System and grading prompts stored in `tools/prompts.py` for easy tuning |

---

## Project Structure

```
rag-weather-agent/
├── agents/
│   └── rag_agent.py          # Main LangGraph agent definition
├── data/
│   └── *.pdf                 # PDF documents to ingest
├── integrations/
│   ├── embeddings.py         # Embedding model helpers
│   ├── langsmith.py          # LangSmith tracing configuration
│   └── qdrant_client.py      # Qdrant vector store client
├── loaders/
│   └── pdf_loader.py         # PDF parsing utilities
├── scripts/
│   ├── ingest_data.py        # CLI script to ingest PDFs into Qdrant
│   └── create_test_pdf.py    # Generates sample PDFs for testing
├── tests/
│   ├── test_graph_flow.py    # Unit tests for the agent graph
│   ├── test_retriever.py     # Retriever tests
│   └── test_tools.py         # Tool-level tests
├── tools/
│   ├── advanced_retriever.py # Sub-graph: retrieve → grade → rewrite loop
│   ├── prompts.py            # All prompt templates
│   ├── retriever.py          # Basic retriever & indexing logic
│   └── weather.py            # OpenWeatherMap integration
├── main.py                   # Interactive CLI entry point
├── requirements.txt          # Python dependencies
├── .env.template             # Template for environment variables
└── README.md                 # This file
```

---

## Prerequisites

- Python 3.10+
- A running **Qdrant** instance (local or cloud)
- API keys for:
  - **OpenAI** (or another LLM provider supported by LangChain)
  - **OpenWeatherMap**
  - **Cohere** (optional, for embeddings)
  - **LangSmith** (optional, for tracing)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/akashxlr8/rag-weather-agent.git
cd rag-weather-agent

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

1. Copy the environment template and fill in your keys:

```bash
cp .env.template .env
```

2. Edit `.env`:

```dotenv
OPENAI_API_KEY=sk-...
OPENWEATHER_API_KEY=...
QDRANT_URL=http://localhost:6333   # or your Qdrant Cloud URL
QDRANT_API_KEY=                     # leave blank for local
COHERE_API_KEY=...                  # if using Cohere embeddings
LANGCHAIN_API_KEY=...               # for LangSmith tracing
LANGCHAIN_PROJECT=rag-weather-agent
```

---

## Usage

### 1. Ingest PDF Documents

Place your PDF files in the `data/` directory, then run:

```bash
python scripts/ingest_data.py
```

This will:
- Parse each PDF
- Chunk text
- Generate embeddings
- Upsert vectors into Qdrant

### 2. Run the Agent (CLI)

```bash
python main.py
```

Example session:

```
Building RAG Agent...
Agent ready! Type 'exit' to quit.
User: What is the weather in Tokyo?
Assistant: Weather in Tokyo: clear sky. Temperature: 18°C. Humidity: 45%. Wind Speed: 3.5 m/s.

User: Who is Akash Kumar Shaw?
Assistant: Akash Kumar Shaw is a Gen AI Developer at TCS working in the BFSI sector... [Source: Akash_Profile]
```

### 3. Run Tests

```bash
python -m unittest discover -s tests
```

---

## Architecture

```
┌─────────────┐
│   User      │
└─────┬───────┘
      │ HumanMessage
      ▼
┌─────────────┐
│  Chatbot    │◄──────────────────────┐
│  (LLM node) │                       │
└─────┬───────┘                       │
      │ tool_calls?                   │
      ▼                               │
┌─────────────┐    ToolMessage        │
│  Tools Node │───────────────────────┘
│  (weather / │
│   retriever)│
└─────────────┘
```

1. **Chatbot Node** – Invokes the LLM with a system prompt and message history. If the LLM requests tools, execution moves to the Tools Node.
2. **Tools Node** – Executes `weather_tool` or `retriever_tool` and returns `ToolMessage`s.
3. **Loop** – After tool execution, control returns to Chatbot for the LLM to synthesize a final answer or request more tools.

### Advanced Retriever Sub-Graph

The `retriever_tool` internally runs a **self-correcting RAG loop**:

```
retrieve → grade_documents
              │
    ┌─────────┴─────────┐
    │ relevant?         │
    ▼ yes               ▼ no (retry < 2)
return_context      rewrite_question
                         │
                         └──► retrieve (loop)
```

---

## Customization

### Prompts

All prompts live in `tools/prompts.py`:

| Constant | Purpose |
|----------|---------|
| `AGENT_SYSTEM_PROMPT` | Main agent persona and instructions |
| `GRADE_PROMPT` | LLM prompt for grading document relevance |
| `REWRITE_PROMPT` | LLM prompt for query rewriting |

### Models

Edit `agents/rag_agent.py` and `tools/advanced_retriever.py` to swap models:

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

---

## LangSmith Tracing

When `LANGCHAIN_API_KEY` is set, all runs are automatically traced. View traces at:

```
https://smith.langchain.com/
```

---
## License

MIT © 2025
