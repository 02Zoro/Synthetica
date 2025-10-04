# Synthetica: How It All Works

## 🎯 Overview

Synthetica is an AI-powered research assistant that generates novel scientific hypotheses by analyzing scientific literature and identifying connections between research concepts.

## 🏗️ System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   AI Agents     │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Multi-Agent) │
│   Port 3000     │    │   Port 8001     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Data Layer   │
                       │   (Vector DB)  │
                       └─────────────────┘
```

## 🔧 Core Components

### 1. Frontend (React Application)
**Location**: `frontend/`
**Port**: 3000

**Key Files**:
- `src/App.js` - Main application component
- `src/pages/ResearchPage.js` - Research query interface
- `src/pages/KnowledgeGraphPage.js` - Knowledge graph visualization
- `src/pages/AboutPage.js` - System information

**Features**:
- Interactive research query form
- Real-time hypothesis generation
- Knowledge graph visualization
- Responsive design

### 2. Backend API (FastAPI)
**Location**: `app/`
**Port**: 8001

**Key Files**:
- `app/simple_demo.py` - Main demo server
- `app/main_demo.py` - Full-featured demo server
- `app/api/routes/research.py` - Research endpoints
- `app/api/routes/health.py` - Health check endpoints

**API Endpoints**:
- `GET /` - Root endpoint with system info
- `GET /health` - Health check
- `POST /research/generate` - Generate hypotheses
- `GET /research/entities/{entity}` - Find related entities
- `GET /research/paths/{entity1}/{entity2}` - Find entity paths

### 3. AI Agents System
**Location**: `app/agents/`

**Agent Types**:
1. **RAG Agent** (`rag_agent.py`)
   - Retrieves relevant scientific papers
   - Uses vector similarity search
   - Filters by domain and relevance

2. **Extraction Agent** (`extraction_agent.py`)
   - Extracts entities (genes, proteins, diseases)
   - Identifies relationships between entities
   - Uses BioBERT for biomedical text processing

3. **Synthesis Agent** (`synthesis_agent.py`)
   - Generates novel hypotheses
   - Combines information from multiple sources
   - Uses GPT-4o for creative synthesis

4. **Critic Agent** (`critic_agent.py`)
   - Reviews and validates hypotheses
   - Identifies potential issues
   - Uses Claude for critical analysis

### 4. Workflow Orchestration
**Location**: `app/agents/workflow.py`

**LangGraph Workflow**:
```
Research Query
      │
      ▼
┌─────────────┐
│ RAG Agent   │ ──► Retrieve Papers
└─────────────┘
      │
      ▼
┌─────────────┐
│ Extraction  │ ──► Extract Entities & Relationships
└─────────────┘
      │
      ▼
┌─────────────┐
│ Synthesis   │ ──► Generate Hypotheses
└─────────────┘
      │
      ▼
┌─────────────┐
│ Critic      │ ──► Review & Validate
└─────────────┘
      │
      ▼
Final Results
```

## 🚀 How It Works: Step by Step

### 1. User Input
- User enters a research question in the frontend
- Selects research domain (biomedical, clinical, etc.)
- Specifies number of hypotheses to generate

### 2. API Request
- Frontend sends POST request to `/research/generate`
- Backend receives query and parameters
- System initializes workflow state

### 3. RAG Agent Processing
- Searches vector database for relevant papers
- Uses semantic similarity to find related research
- Retrieves abstracts and metadata

### 4. Entity Extraction
- BioBERT model processes retrieved papers
- Extracts named entities (genes, proteins, diseases)
- Identifies relationships between entities
- Builds knowledge graph structure

### 5. Hypothesis Synthesis
- GPT-4o analyzes extracted information
- Generates novel research hypotheses
- Combines insights from multiple papers
- Creates testable predictions

### 6. Critical Review
- Claude reviews generated hypotheses
- Identifies potential issues or gaps
- Suggests improvements or alternatives
- Validates scientific soundness

### 7. Results Presentation
- Formatted hypotheses returned to frontend
- Includes confidence scores and methodology
- Shows processing time and metadata
- Displays in user-friendly interface

## 🎭 Demo Mode vs Production Mode

### Demo Mode (Current Setup)
**Configuration**: `app/core/config_demo.py`
**Features**:
- Mock API keys (no real LLM calls)
- Predefined responses for common queries
- Fast response times
- No external dependencies

**Mock Responses**:
- Cancer queries → BRCA1/breast cancer hypotheses
- Diabetes queries → Metabolic reprogramming hypotheses
- Alzheimer's queries → Synaptic dysfunction hypotheses
- Generic queries → Interdisciplinary research hypotheses

### Production Mode
**Configuration**: `app/core/config.py`
**Features**:
- Real API keys for OpenAI, Anthropic
- Actual LLM processing
- Vector database integration
- Knowledge graph construction

## 📊 Data Flow

### 1. Query Processing
```
User Query → Frontend → Backend API → Workflow Orchestrator
```

### 2. Agent Execution
```
Workflow → RAG Agent → Extraction Agent → Synthesis Agent → Critic Agent
```

### 3. Response Generation
```
Agent Results → Response Formatter → Frontend Display
```

## 🔍 Key Technologies

### Backend Technologies
- **FastAPI**: High-performance web framework
- **LangGraph**: Multi-agent workflow orchestration
- **Pydantic**: Data validation and serialization
- **Structlog**: Structured logging

### Frontend Technologies
- **React**: User interface library
- **Ant Design**: UI component library
- **Axios**: HTTP client for API calls
- **Styled Components**: CSS-in-JS styling

### AI/ML Technologies
- **OpenAI GPT-4o**: Hypothesis generation
- **Anthropic Claude**: Critical review
- **BioBERT**: Biomedical text processing
- **Sentence Transformers**: Text embeddings

### Data Storage
- **ChromaDB**: Vector database for embeddings
- **Neo4j**: Graph database for knowledge graphs
- **Redis**: Caching and session storage

## 🚀 Getting Started

### 1. Backend Startup
```bash
# Start the demo backend
python -m uvicorn app.simple_demo:app --reload --host 0.0.0.0 --port 8001
```

### 2. Frontend Startup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### 3. Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## 🧪 Testing the System

### 1. Health Check
```bash
curl http://localhost:8001/health
```

### 2. Generate Hypotheses
```bash
curl -X POST http://localhost:8001/research/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes cancer?", "domain": "biomedical", "max_hypotheses": 3}'
```

### 3. Frontend Testing
- Open http://localhost:3000
- Enter a research question
- Click "Generate Hypotheses"
- View results

## 🔧 Configuration

### Environment Variables
```bash
# Demo Mode
DEMO_MODE=true
DEBUG=true

# API Keys (Production)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database URLs
NEO4J_URI=bolt://localhost:7687
REDIS_URL=redis://localhost:6379/0
```

### API Configuration
- **CORS**: Enabled for localhost:3000
- **Rate Limiting**: Not implemented in demo
- **Authentication**: Not required for demo

## 📈 Performance Characteristics

### Demo Mode Performance
- **Response Time**: 1-2 seconds
- **Memory Usage**: ~100MB
- **CPU Usage**: Low (mock responses)

### Production Mode Performance
- **Response Time**: 10-30 seconds
- **Memory Usage**: ~500MB
- **CPU Usage**: High (LLM processing)

## 🛠️ Troubleshooting

### Common Issues

1. **Frontend won't connect to backend**
   - Check if backend is running on port 8001
   - Verify API endpoint URLs in frontend code

2. **Backend won't start**
   - Check Python dependencies are installed
   - Verify no port conflicts

3. **Frontend won't start**
   - Run `npm install` in frontend directory
   - Check Node.js version compatibility

### Debug Commands
```bash
# Check backend health
curl http://localhost:8001/health

# Check frontend
curl http://localhost:3000

# View logs
# Backend logs appear in terminal
# Frontend logs in browser console
```

## 🎯 Use Cases

### 1. Research Discovery
- Generate novel research hypotheses
- Identify research gaps
- Explore interdisciplinary connections

### 2. Literature Analysis
- Analyze scientific papers
- Extract key insights
- Build knowledge graphs

### 3. Hypothesis Testing
- Validate research ideas
- Identify potential issues
- Suggest improvements

## 🔮 Future Enhancements

### Planned Features
- Real-time collaboration
- Advanced visualization
- Integration with research databases
- Automated paper analysis
- Multi-language support

### Technical Improvements
- Performance optimization
- Scalability improvements
- Enhanced security
- Better error handling

## 📚 Additional Resources

- **Architecture Documentation**: `ARCHITECTURE.md`
- **Setup Guide**: `SETUP.md`
- **Manual Startup**: `MANUAL_STARTUP.md`
- **Demo Mode Guide**: `DEMO_MODE.md`

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

### Code Structure
- Follow existing patterns
- Add comprehensive tests
- Update documentation
- Maintain backward compatibility

---

**SABDE** - Transforming scientific research through AI-powered hypothesis generation! 🚀
