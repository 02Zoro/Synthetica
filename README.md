# Synthetica - AI-Powered Scientific Research Assistant

An AI-powered scientific research assistant that uses multi-agent systems, knowledge graphs, and advanced NLP to generate novel research hypotheses from scientific literature.

## Features

- **Multi-Agent Architecture**: Specialized agents for data retrieval, extraction, synthesis, and validation
- **Advanced RAG Pipeline**: Domain-specific embeddings with BioBERT and semantic search
- **Knowledge Graph Integration**: Neo4j for structured scientific relationships
- **Self-Correction Loop**: Generator-Critic pattern for hypothesis validation
- **Interactive Visualization**: React frontend with graph visualization capabilities

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI     │    │   FastAPI       │    │   LangGraph     │
│   (Frontend)    │◄──►│   (API Layer)   │◄──►│   (Orchestration)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Vector DB     │    │   Knowledge     │
                       │   (ChromaDB)    │    │   Graph (Neo4j) │
                       └─────────────────┘    └─────────────────┘
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start the Services**
   ```bash
   # Start the FastAPI backend
   uvicorn app.main:app --reload
   
   # Start the frontend (in another terminal)
   cd frontend && npm start
   ```

## Project Structure

```
synthetica/
├── app/                    # FastAPI application
│   ├── agents/            # LangGraph agents
│   ├── services/          # Business logic services
│   ├── models/            # Pydantic models
│   └── main.py           # FastAPI app entry point
├── frontend/              # React frontend
├── data/                  # Sample scientific papers
├── notebooks/             # Jupyter notebooks for analysis
└── tests/                 # Test files
```

## Technology Stack

- **Orchestration**: LangGraph, LangChain
- **API Layer**: FastAPI
- **Vector Database**: ChromaDB (local), Pinecone (production)
- **Knowledge Graph**: Neo4j
- **LLMs**: GPT-4o, Claude
- **Frontend**: React, Cytoscape.js
- **Monitoring**: Prometheus, Grafana

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black app/
flake8 app/
```

## License

MIT License
