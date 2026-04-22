# Grid07: Cognitive Routing & RAG

An AI-driven orchestration system that routes content to persona-based bots, enriches responses with real-time context, and defends against prompt injection attacks.

## Features

- **🎯 Semantic Routing**: Vector-based routing to 3 distinct bot personas using FAISS
- **🔍 RAG Pipeline**: LangGraph-powered workflow with web search integration
- **🛡️ Defense Layer**: Keyword-based jailbreak detection
- **⚡ Production Ready**: Pydantic validation, error handling, logging

## Architecture

The system consists of 3 phases:

### Phase 1: Routing
Routes incoming queries to the most relevant bot persona using vector similarity.

### Phase 2: Generation  
LangGraph pipeline that decides search topics, retrieves context, and generates responses.

### Phase 3: Defense
Detects and blocks malicious inputs attempting prompt injection.

See [docs/architecture.md](docs/architecture.md) for detailed architecture.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Usage

```python
from src.routing import VectorRouter
from src.generation import build_generation_graph
from src.defense import DefenseEngine

# Phase 1: Route query
router = VectorRouter()
bots = router.route_post_to_bots("Your query here")

# Phase 2: Generate response
graph = build_generation_graph()
result = graph.invoke(state_dict)

# Phase 3: Check for jailbreak
defense = DefenseEngine()
defense_check = defense.process_with_defense(state)
```

## Project Structure

```
src/
├── routing.py          # Phase 1: Vector-based routing
├── generation.py       # Phase 2: LangGraph pipeline
├── defense.py          # Phase 3: Jailbreak detection
├── models.py           # Pydantic data models
├── config.py           # Configuration & constants
└── __init__.py

docs/
├── architecture.md     # System architecture
└── README.md          # This file

test_phases.py         # Integrated testing script
test_defense.py        # Defense layer tests
Grid07_Complete.ipynb  # Jupyter notebook showcase
```

## Configuration

Key settings in `src/config.py`:

- `SIMILARITY_THRESHOLD`: 0.65 (routing threshold)
- `MAX_CONVERSATION_HISTORY`: 5 (FIFO buffer)
- `EMBEDDING_MODEL`: sentence-transformers/all-MiniLM-L6-v2
- `LLM_MODEL`: llama-3.1-8b-instant (via Groq)

## Testing

Run all phases:
```bash
python test_phases.py
```

Test defense layer:
```bash
python test_defense.py
```

## Bot Personas

- **bot_A**: Optimistic tech enthusiast (AI/crypto advocate)
- **bot_B**: Critical privacy advocate (monopoly opponent)
- **bot_C**: Finance-focused algorithmic trader

## Dependencies

- `sentence-transformers`: Vector embeddings
- `faiss-cpu`: Vector similarity search
- `langchain-groq`: LLM integration
- `langgraph`: Workflow orchestration
- `pydantic`: Data validation
- `duckduckgo-search`: Real-time web search

## License

Educational project for Internshala assignment.
