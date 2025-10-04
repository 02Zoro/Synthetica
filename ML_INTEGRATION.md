# SABDE ML Integration Guide

## 🧠 ML Pipeline Integration Complete!

Your SABDE system now includes the advanced ML pipeline from `Synthetica-AI__ML-` with the following capabilities:

### **✅ What's Been Integrated**

1. **Graph Neural Network (GNN) Hypothesis Generation**
   - Advanced ML models for scientific hypothesis generation
   - Graph-based analysis of scientific relationships
   - Novel connection discovery

2. **Knowledge Graph Construction**
   - Neo4j-based scientific knowledge graphs
   - Entity relationship extraction
   - Scientific paper analysis

3. **Vector Database Integration**
   - ChromaDB for semantic search
   - Scientific paper embeddings
   - Similarity-based retrieval

4. **Enhanced API Endpoints**
   - ML-powered hypothesis generation
   - Real-time ML pipeline status
   - Advanced research capabilities

## 🚀 **How to Use the ML-Enhanced System**

### **Option 1: Quick Start (Demo Mode)**
```bash
# Start the enhanced demo (works without ML dependencies)
python scripts/start_demo.py
```

### **Option 2: Full ML Pipeline**
```bash
# Install ML dependencies first
pip install -r requirements-ml.txt

# Start the ML-enhanced system
python scripts/start_ml.py
```

### **Option 3: Manual ML Setup**
```bash
# Set environment variables
export ML_PIPELINE_ENABLED=true
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password

# Start the application
python -m uvicorn app.main_ml:app --reload --host 0.0.0.0 --port 8001
```

## 📊 **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Pipeline  │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (GNN + KG)   │
│   Port 3000     │    │   Port 8001     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Data Layer   │
                       │   (Neo4j + VDB) │
                       └─────────────────┘
```

## 🔧 **New API Endpoints**

### **ML-Enhanced Research**
- `POST /api/v1/research/generate` - ML-powered hypothesis generation
- `GET /api/v1/research/ml-status` - ML pipeline status
- `POST /api/v1/research/ml-initialize` - Initialize ML pipeline

### **System Status**
- `GET /api/v1/status` - Comprehensive system status
- `GET /` - Root endpoint with ML capabilities info

## 🧬 **ML Pipeline Components**

### **1. Graph Neural Network (GNN)**
- **File**: `app/ml_pipeline/gnn_hypothesis_generator.py`
- **Purpose**: Generate novel scientific hypotheses
- **Features**: 
  - Node embedding learning
  - Link prediction
  - Novel connection discovery

### **2. Knowledge Graph Builder**
- **File**: `app/ml_pipeline/kg_builder.py`
- **Purpose**: Extract entities and relationships from scientific papers
- **Features**:
  - Named Entity Recognition (NER)
  - Relationship extraction
  - Neo4j graph construction

### **3. Vector Database**
- **File**: `app/ml_pipeline/vectorize_and_index.py`
- **Purpose**: Semantic search and similarity matching
- **Features**:
  - Scientific paper embeddings
  - Semantic similarity search
  - Evidence-based hypothesis grounding

### **4. Data Collection**
- **File**: `app/ml_pipeline/script.py`
- **Purpose**: Collect scientific papers from PubMed
- **Features**:
  - Automated paper collection
  - Abstract extraction
  - Data preprocessing

## 🎯 **Usage Examples**

### **1. Check ML Pipeline Status**
```bash
curl http://localhost:8001/api/v1/research/ml-status
```

### **2. Generate ML-Enhanced Hypotheses**
```bash
curl -X POST http://localhost:8001/api/v1/research/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes cancer?", "domain": "biomedical", "max_hypotheses": 3}'
```

### **3. Initialize ML Pipeline**
```bash
curl -X POST http://localhost:8001/api/v1/research/ml-initialize
```

## 🔄 **Fallback Behavior**

The system is designed to gracefully fall back to demo mode if:
- ML dependencies are not installed
- Neo4j database is not available
- Vector database is not accessible
- ML pipeline initialization fails

## 📈 **Performance Characteristics**

### **Demo Mode**
- **Response Time**: 1-2 seconds
- **Memory Usage**: ~100MB
- **Dependencies**: Minimal

### **ML-Enhanced Mode**
- **Response Time**: 10-30 seconds (first run), 5-10 seconds (subsequent)
- **Memory Usage**: ~1-2GB
- **Dependencies**: Full ML stack

## 🛠️ **Configuration**

### **Environment Variables**
```bash
# ML Pipeline Settings
ML_PIPELINE_ENABLED=true
DEMO_MODE=false

# Database Settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Vector Database
CHROMA_DB_PATH=./app/ml_pipeline/chroma_db_gene_mvp
```

### **Dependencies**
- **Core**: `requirements.txt` (basic functionality)
- **ML-Enhanced**: `requirements-ml.txt` (full ML capabilities)

## 🚨 **Troubleshooting**

### **Common Issues**

1. **ML Pipeline Not Initializing**
   - Check Neo4j connection
   - Verify vector database exists
   - Check ML dependencies

2. **Slow Performance**
   - First run takes longer (model training)
   - Subsequent runs are faster
   - Consider using demo mode for testing

3. **Memory Issues**
   - ML pipeline requires more memory
   - Consider using demo mode for development

### **Debug Commands**
```bash
# Check ML pipeline status
curl http://localhost:8001/api/v1/research/ml-status

# Check system status
curl http://localhost:8001/api/v1/status

# View logs
tail -f logs/sabde.log
```

## 🎉 **What You Can Do Now**

### **1. Test ML-Enhanced Hypotheses**
- Generate hypotheses using Graph Neural Networks
- Get evidence-based predictions
- Explore novel scientific connections

### **2. Explore Knowledge Graphs**
- Visualize scientific relationships
- Find entity connections
- Discover research pathways

### **3. Use Advanced Search**
- Semantic similarity search
- Vector-based retrieval
- Context-aware results

## 🔮 **Next Steps**

1. **Install ML Dependencies**: `pip install -r requirements-ml.txt`
2. **Start ML-Enhanced System**: `python scripts/start_ml.py`
3. **Test ML Capabilities**: Use the enhanced API endpoints
4. **Explore Advanced Features**: Knowledge graphs, vector search, GNN predictions

Your SABDE system now has both educational demo capabilities AND advanced ML capabilities! 🚀

