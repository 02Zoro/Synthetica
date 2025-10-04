# SABDE Demo Mode - No API Keys Required

This guide shows you how to run SABDE in demo mode without needing API keys for OpenAI or Anthropic.

## 🎯 What is Demo Mode?

Demo mode allows you to:
- ✅ **Test the complete system architecture**
- ✅ **See how the multi-agent workflow works**
- ✅ **Experience the user interface**
- ✅ **Understand the data flow**
- ✅ **No API keys or external services required**

## 🚀 Quick Start (Demo Mode)

### **Option 1: One-Command Demo**
```powershell
# Copy demo environment
Copy-Item env.demo .env

# Start demo backend
python scripts/start_demo.py
```

### **Option 2: Manual Demo Setup**
```powershell
# 1. Set demo mode
$env:DEMO_MODE = "true"

# 2. Start demo backend
python -m uvicorn app.main_demo:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 How Demo Mode Works

### **Mock Data Instead of Real AI**
- **RAG Agent**: Returns pre-defined scientific papers
- **Extraction Agent**: Returns mock entities and relationships
- **Synthesis Agent**: Returns realistic research hypotheses
- **Critic Agent**: Returns mock critiques and validation

### **Sample Demo Data**
The demo includes realistic scientific content:
- **Papers**: BRCA1 mutations, diabetes research, Alzheimer's disease
- **Entities**: Genes, proteins, diseases, drugs
- **Hypotheses**: Novel therapeutic targets, metabolic mechanisms
- **Critiques**: Scientific validation and recommendations

## 📊 Demo Features

### **1. Research Hypothesis Generation**
- Enter any research question
- Get realistic hypotheses based on your query
- See testable predictions and methodology
- View confidence scores and validation

### **2. Knowledge Graph Exploration**
- Search for related entities
- Find paths between concepts
- Explore scientific relationships
- Interactive graph visualization

### **3. Multi-Agent Workflow**
- See how agents work together
- Understand the self-correction loop
- Monitor agent execution
- View workflow status

## 🎮 Demo Examples

### **Example 1: Cancer Research**
```
Query: "What are the molecular mechanisms of breast cancer?"
Result: Hypothesis about BRCA1 mutations and therapeutic targets
```

### **Example 2: Diabetes Research**
```
Query: "How does insulin resistance develop?"
Result: Hypothesis about mitochondrial dysfunction and metabolic reprogramming
```

### **Example 3: Alzheimer's Research**
```
Query: "What causes cognitive decline in Alzheimer's disease?"
Result: Hypothesis about synaptic dysfunction and protein interactions
```

## 🔍 Testing the System

### **1. Backend API Testing**
```powershell
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test research endpoint
curl -X POST http://localhost:8000/api/v1/research/generate -H "Content-Type: application/json" -d '{"query": "What causes cancer?", "domain": "biomedical", "max_hypotheses": 3}'
```

### **2. Frontend Testing**
```powershell
# Start frontend (in new terminal)
cd frontend
npm install
npm start
```

### **3. API Documentation**
Visit: http://localhost:8000/docs

## 🛠️ Demo Configuration

### **Environment Variables**
```env
DEMO_MODE=true
DEBUG=true
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///./demo.db
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### **Demo Data Sources**
- **Scientific Papers**: Real abstracts from biomedical literature
- **Entities**: Genes, proteins, diseases, drugs
- **Relationships**: Biological interactions and associations
- **Hypotheses**: Realistic research questions and predictions

## 📈 What You Can Learn

### **1. System Architecture**
- Multi-agent orchestration with LangGraph
- Vector database operations with ChromaDB
- Knowledge graph construction with Neo4j
- API design with FastAPI

### **2. AI Workflow**
- Document retrieval and processing
- Entity extraction and relationship mapping
- Hypothesis generation and validation
- Self-correction and refinement

### **3. User Experience**
- Research query interface
- Hypothesis visualization
- Knowledge graph exploration
- Real-time workflow monitoring

## 🔄 Demo vs Production

| Feature | Demo Mode | Production Mode |
|---------|-----------|-----------------|
| **API Keys** | ❌ Not required | ✅ Required |
| **Real AI** | ❌ Mock responses | ✅ GPT-4o, Claude |
| **Data Sources** | ❌ Pre-defined | ✅ Live databases |
| **Cost** | ❌ Free | ✅ API usage costs |
| **Performance** | ⚡ Fast | ⏱️ Depends on API |
| **Accuracy** | 📝 Realistic | 🎯 High accuracy |

## 🎯 Demo Use Cases

### **1. Educational Purposes**
- Learn about AI research systems
- Understand multi-agent architectures
- Explore scientific hypothesis generation
- Study knowledge graph applications

### **2. System Testing**
- Test the user interface
- Validate the API endpoints
- Check the data flow
- Verify the workflow logic

### **3. Proof of Concept**
- Demonstrate the system capabilities
- Show the user experience
- Validate the architecture
- Test the integration

## 🚀 Next Steps

### **To Use Real AI:**
1. Get API keys from OpenAI and Anthropic
2. Set `DEMO_MODE=false` in environment
3. Use the full production system

### **To Deploy:**
1. Set up production databases
2. Configure cloud services
3. Deploy with Docker
4. Set up monitoring

## 🆘 Troubleshooting

### **Common Issues**
```powershell
# If demo doesn't start
python -c "import app.main_demo; print('Demo module OK')"

# If dependencies missing
pip install -r requirements-simple.txt

# If port in use
netstat -ano | findstr :8000
```

### **Demo Limitations**
- Mock data only (not real AI)
- Limited to pre-defined responses
- No real-time learning
- Simplified workflow

## 📚 Additional Resources

- **Architecture Guide**: `ARCHITECTURE.md`
- **Manual Setup**: `MANUAL_STARTUP.md`
- **API Documentation**: http://localhost:8000/docs
- **Source Code**: `app/agents/demo_agents.py`

## 🎉 Enjoy the Demo!

The demo mode gives you a complete understanding of how SABDE works without any external dependencies. You can explore the system, test the interface, and understand the architecture before deciding to use the full AI-powered version.

**Happy exploring! 🚀**
