# Synthetica Manual Startup Guide

This guide provides step-by-step instructions for manually setting up and running the Synthetica system with both demo mode and ML-enhanced capabilities.

## 🎯 System Modes

Synthetica now supports three modes:

1. **Demo Mode** - Educational demonstration without ML dependencies
2. **ML-Enhanced Mode** - Full AI/ML pipeline with Graph Neural Networks and Knowledge Graphs
3. **Real Data Mode** - Full system with your actual biomedical data (15,183 nodes, 10,025 edges)

## 🚀 Quick Start

### Demo Mode (Fastest Setup)

#### PowerShell (Windows):
```powershell
# 1. Install basic dependencies
pip install -r requirements.txt

# 2. Start demo backend
uvicorn app.simple_demo:app --reload --host 0.0.0.0 --port 8001

# 3. Start frontend (in new terminal)
cd frontend; npm install; npm start

# 4. Open http://localhost:3000
```

#### Bash (Linux/Mac):
```bash
# 1. Install basic dependencies
pip install -r requirements.txt

# 2. Start demo backend
uvicorn app.simple_demo:app --reload --host 0.0.0.0 --port 8001

# 3. Start frontend (in new terminal)
cd frontend && npm install && npm start

# 4. Open http://localhost:3000
```

### ML-Enhanced Mode (Full Capabilities)

#### PowerShell (Windows):
```powershell
# 1. Install ML dependencies
pip install -r requirements-ml.txt

# 2. Setup databases (Neo4j, PostgreSQL)
docker run --name sabde-neo4j -p 7687:7687 -e NEO4J_AUTH=neo4j/password -d neo4j:5.0-community

# 3. Start ML-enhanced backend
python -m uvicorn app.main_ml:app --reload --host 0.0.0.0 --port 8001

# 4. Start frontend (in new terminal)
cd frontend; npm install; npm start

# 5. Open http://localhost:3000
```

#### Bash (Linux/Mac):
```bash
# 1. Install ML dependencies
pip install -r requirements-ml.txt

# Start Docker Desktop first, then:
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/synthetica_password neo4j:latest

# 2. Setup databases (Neo4j, PostgreSQL)
docker run --name sabde-neo4j -p 7687:7687 -e NEO4J_AUTH=neo4j/password -d neo4j:5.0-community

# 3. Start ML-enhanced backend
uvicorn app.main_ml:app --reload --host 0.0.0.0 --port 8001

# 4. Start frontend (in new terminal)
cd frontend && npm install && npm start

# 5. Open http://localhost:3000
```

### Real Data Mode (Your Biomedical Data)

#### PowerShell (Windows):
```powershell
# 1. Install ML dependencies
pip install -r requirements-ml.txt

# 2. Start Neo4j with your real data
docker run -d --name synthetica-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/synthetica_password neo4j:latest

# 3. Wait for Neo4j to start (30 seconds)
Start-Sleep -Seconds 30

# 4. Populate Neo4j with your real gene data
python populate_neo4j_with_real_data.py

# 5. Start ML-enhanced backend with real data
python -m uvicorn app.main_ml:app --reload --host 0.0.0.0 --port 8001

# 6. Start frontend (in new terminal)
cd frontend; npm install; npm start

# 7. Open http://localhost:3000
```

#### Bash (Linux/Mac):
```bash
# 1. Install ML dependencies
pip install -r requirements-ml.txt

# 2. Start Neo4j with your real data
docker run -d --name synthetica-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/synthetica_password neo4j:latest

# 3. Wait for Neo4j to start (30 seconds)
sleep 30

# 4. Populate Neo4j with your real gene data
python populate_neo4j_with_real_data.py

# 5. Start ML-enhanced backend with real data
uvicorn app.main_ml:app --reload --host 0.0.0.0 --port 8001

# 6. Start frontend (in new terminal)
cd frontend && npm install && npm start

# 7. Open http://localhost:3000
```

## Prerequisites

Before starting, ensure you have the following installed:

### Required Software

1. **Python 3.11+**
   ```bash
   python --version
   # Should show Python 3.11 or higher
   ```

2. **Node.js 18+**
   ```bash
   node --version
   # Should show v18 or higher
   ```

3. **Git**
   ```bash
   git --version
   ```

### Optional (for production)
- **Docker and Docker Compose**
- **PostgreSQL** (if not using Docker)
- **Neo4j** (if not using Docker)
- **Redis** (if not using Docker)

## Step 1: Clone and Setup Project

### 1.1 Clone Repository
```bash
git clone <repository-url>
cd sabde
```

### 1.2 Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 1.3 Install Python Dependencies

#### Option A: Demo Mode (Basic Dependencies)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: ML-Enhanced Mode (Full ML Dependencies)
```bash
pip install --upgrade pip
pip install -r requirements-ml.txt
```

**Note**: ML-Enhanced mode requires additional dependencies for Graph Neural Networks, Knowledge Graphs, and Vector Databases.

## Step 2: Database Setup

### 2.1 PostgreSQL Setup (Optional for Demo Mode)

**Note**: PostgreSQL is optional for demo mode but required for ML-enhanced mode.

#### Option A: Using Docker
```bash
docker run --name sabde-postgres \
  -e POSTGRES_DB=sabde \
  -e POSTGRES_USER=sabde_user \
  -e POSTGRES_PASSWORD=sabde_password \
  -p 5432:5432 \
  -d postgres:15
```

#### Option B: Local Installation
1. Install PostgreSQL from https://www.postgresql.org/download/
2. Create database and user:
```sql
CREATE DATABASE sabde;
CREATE USER sabde_user WITH PASSWORD 'sabde_password';
GRANT ALL PRIVILEGES ON DATABASE sabde TO sabde_user;
```

### 2.2 Neo4j Setup (Required for ML-Enhanced Mode)

**Note**: Neo4j is required for ML-enhanced mode to store knowledge graphs.

#### Option A: Using Docker
```bash
docker run --name sabde-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/sabde_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -d neo4j:5.0-community
```

#### Option B: Local Installation
1. Download Neo4j from https://neo4j.com/download/
2. Install and start Neo4j
3. Set password to `sabde_password`
4. Install APOC plugin

### 2.3 Redis Setup (Optional)

#### Option A: Using Docker
```bash
docker run --name sabde-redis \
  -p 6379:6379 \
  -d redis:7-alpine
```

#### Option B: Local Installation
```bash
# On Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# On macOS
brew install redis
brew services start redis

# On Windows
# Download from https://github.com/microsoftarchive/redis/releases
```

## Step 3: Environment Configuration

### 3.1 Create Environment File
```bash
cp env.example .env
```

### 3.2 Configure Environment Variables

#### Option A: Demo Mode Configuration
Create `.env` file for demo mode:
```env
# Demo Mode Settings
DEMO_MODE=true
DEBUG=true
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8001

# Optional: Mock API keys for demo
OPENAI_API_KEY=sk-demo-openai-key
ANTHROPIC_API_KEY=sk-demo-anthropic-key
```

#### Option B: ML-Enhanced Mode Configuration
Create `.env` file for ML-enhanced mode:
```env
# ML Pipeline Settings
ML_PIPELINE_ENABLED=true
DEMO_MODE=false

# API Keys (Required for ML mode)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
DATABASE_URL=postgresql://sabde_user:sabde_password@localhost:5432/sabde
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=sabde_password

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_DB_PATH=./app/ml_pipeline/chroma_db_gene_mvp

# Redis (Optional)
REDIS_URL=redis://localhost:6379

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8001
SECRET_KEY=your_secret_key_here
```

#### Option C: Real Data Mode Configuration
Create `.env` file for real data mode:
```env
# Real Data Mode Settings
ML_PIPELINE_ENABLED=true
ML_MODE=true
DEMO_MODE=false

# API Keys (Required for real data mode)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=synthetica_password

# Vector Database (Your real gene data)
CHROMA_DB_PATH=./app/ml_pipeline/chroma_db_gene_mvp_new

# ML Pipeline Settings
GNN_MODEL_PATH=./app/ml_pipeline/models
KNOWLEDGE_GRAPH_PATH=./app/ml_pipeline/kg_data

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8001
SECRET_KEY=your_secret_key_here
```

### 3.3 Get API Keys

#### OpenAI API Key
1. Go to https://platform.openai.com/
2. Sign up/Login
3. Go to API Keys section
4. Create new API key
5. Copy and paste into `.env` file

#### Anthropic API Key
1. Go to https://console.anthropic.com/
2. Sign up/Login
3. Go to API Keys section
4. Create new API key
5. Copy and paste into `.env` file

## Step 4: Initialize Data

### 4.1 Create Data Directory
```bash
mkdir -p chroma_db
```

### 4.2 Initialize Sample Data
```bash
python scripts/init_data.py
```

This will:
- Load sample scientific papers into the vector database
- Initialize the knowledge graph with sample entities
- Set up the database schema

## Step 5: Start Backend Services

### 5.1 Start FastAPI Backend

#### Option A: Demo Mode
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Start demo backend
uvicorn app.simple_demo:app --reload --host 0.0.0.0 --port 8001
```

#### Option B: ML-Enhanced Mode
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Start ML-enhanced backend
uvicorn app.main_ml:app --reload --host 0.0.0.0 --port 8001
```

#### Option C: Original Full-Featured Mode
```bash
# Start original backend (requires all dependencies)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.2 Verify Backend is Running
Open a new terminal and test the API:

#### For Demo Mode (Port 8001):
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{"status": "healthy", "service": "sabde-demo"}
```

#### For ML-Enhanced Mode (Port 8001):
```bash
curl http://localhost:8001/api/v1/health
```

Expected response:
```json
{"status": "healthy", "service": "sabde"}
```

#### Check ML Pipeline Status (ML-Enhanced Mode):
```bash
curl http://localhost:8001/api/v1/research/ml-status
```

## Step 6: Setup Frontend

### 6.1 Navigate to Frontend Directory
```bash
cd frontend
```

### 6.2 Install Node.js Dependencies
```bash
npm install
```

### 6.3 Start React Development Server
```bash
npm start
```

You should see output like:
```
Compiled successfully!

You can now view sabde-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.100:3000

Note that the development build is not optimized.
To create a production build, use npm run build.
```

## Step 7: Verify Installation

### 7.1 Check All Services

#### Backend API
- **Demo Mode**: http://localhost:8001
- **ML-Enhanced Mode**: http://localhost:8001
- **Original Mode**: http://localhost:8000
- Health Check: `/health` or `/api/v1/health`
- API Docs: `/docs`

#### Frontend
- URL: http://localhost:3000
- Should show the SABDE homepage

#### ML Pipeline Status (ML-Enhanced Mode)
- ML Status: http://localhost:8001/api/v1/research/ml-status
- System Status: http://localhost:8001/api/v1/status

#### Database Connections
```bash
# Test PostgreSQL connection
psql -h localhost -U sabde_user -d sabde -c "SELECT 1;"

# Test Neo4j connection (if using Docker)
docker exec sabde-neo4j cypher-shell -u neo4j -p sabde_password "RETURN 1;"
```

### 7.2 Test Research Functionality

#### Demo Mode Testing
1. Open http://localhost:3000 in your browser
2. Navigate to the Research page
3. Enter a test query like: "What are the molecular mechanisms of Alzheimer's disease?"
4. Select domain: "Biomedical"
5. Click "Generate Hypotheses"
6. Wait for results (should be fast, 1-2 seconds)

#### ML-Enhanced Mode Testing
1. Open http://localhost:3000 in your browser
2. Navigate to the Research page
3. Enter a test query like: "What are the molecular mechanisms of Alzheimer's disease?"
4. Select domain: "Biomedical"
5. Click "Generate Hypotheses"
6. Wait for results (may take 10-30 seconds for first run, faster for subsequent runs)

#### Real Data Mode Testing
1. Open http://localhost:3000 in your browser
2. Navigate to the Research page
3. Enter a test query like: "BRCA1 mutations and breast cancer treatment"
4. Select domain: "Biomedical"
5. Click "Generate Hypotheses"
6. Wait for results (should show real gene names and disease terms)
7. Check that the response shows `entities > 0` and `relationships > 0`

#### Test ML Pipeline Status
```bash
# Check if ML pipeline is working
curl http://localhost:8001/api/v1/research/ml-status

# Test ML hypothesis generation
curl -X POST http://localhost:8001/api/v1/research/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes cancer?", "domain": "biomedical", "max_hypotheses": 3}'

# Test with real data (should show entities > 0)
curl -X POST http://localhost:8001/api/v1/research/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "BRCA1 mutations and breast cancer", "domain": "biomedical", "max_hypotheses": 2}'
```

## Step 8: ML Pipeline Setup (ML-Enhanced Mode Only)

### 8.1 Initialize ML Pipeline Data

#### Option A: Use Pre-built Data
If you have the ML pipeline data from `Synthetica-AI__ML-`:
```bash
# Copy ML pipeline data
cp -r Synthetica-AI__ML-/chroma_db_gene_mvp app/ml_pipeline/
cp Synthetica-AI__ML-/*.py app/ml_pipeline/
```

#### Option B: Build ML Pipeline from Scratch
```bash
# Navigate to ML pipeline directory
cd app/ml_pipeline

# Collect scientific papers (requires API key)
python script.py

# Build knowledge graph
python kg_builder.py

# Create vector database
python vectorize_and_index.py

# Test ML pipeline
python gnn_hypothesis_generator.py
```

### 8.2 Initialize ML Pipeline in Application
```bash
# Initialize ML pipeline
curl -X POST http://localhost:8001/api/v1/research/ml-initialize

# Check ML pipeline status
curl http://localhost:8001/api/v1/research/ml-status
```

## Step 8.5: Real Data Mode Setup (Your Biomedical Data)

### 8.5.1 Prepare Your Real Data
```bash
# Ensure you have your ChromaDB data
ls -la app/ml_pipeline/chroma_db_gene_mvp_new/

# Check if Neo4j is running
docker ps | grep neo4j
```

### 8.5.2 Populate Neo4j with Real Data
```bash
# Populate Neo4j with your 15,082 scientific abstracts
python populate_neo4j_with_real_data.py

# Verify data was loaded
python check_neo4j_stats.py
```

Expected output:
```
Neo4j Database Statistics:
Total Nodes: 15183
Total Edges: 10025

Node Types:
  ['Document']: 15082
  ['Gene']: 57
  ['Disease']: 44

Edge Types:
  DISCUSSES: 8419
  MENTIONS: 1606
```

### 8.5.3 Test Real Data Mode
```bash
# Test API with real data
curl -X POST http://localhost:8001/api/v1/research/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "BRCA1 mutations and breast cancer", "domain": "biomedical", "max_hypotheses": 2}'
```

Expected response should show:
- `entities > 0` (should be 15183)
- `relationships > 0` (should be 10025)
- Real gene names and disease terms in the hypotheses

## Step 9: Optional - Setup Monitoring

### 9.1 Start Monitoring Services
```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### 9.2 Access Monitoring Dashboards
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

## Troubleshooting

### Common Issues and Solutions

#### 1. PowerShell Command Chaining Issues
**Error**: `The token '&&' is not a valid statement separator in this version`

**Solutions**:
```powershell
# Use semicolon (;) instead of && for PowerShell
cd frontend; npm install; npm start

# Or use separate commands
cd frontend
npm install
npm start

# Or use PowerShell's -and operator for conditional execution
cd frontend -and npm install -and npm start
```

#### 2. Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port
netstat -tulpn | grep :8000
# or
lsof -i :8000

# Kill the process
kill -9 <PID>
```

#### 2. Database Connection Failed
**Error**: `Connection refused` or `Authentication failed`

**Solutions**:
- Check if databases are running
- Verify connection strings in `.env`
- Check firewall settings
- Ensure correct credentials

#### 3. API Key Errors
**Error**: `Invalid API key` or `Rate limit exceeded`

**Solutions**:
- Verify API keys in `.env` file
- Check API key quotas and billing
- Ensure keys have proper permissions

#### 4. Module Import Errors
**Error**: `ModuleNotFoundError`

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
echo $PYTHONPATH
```

#### 5. Frontend Build Errors
**Error**: `npm ERR!` or `Module not found`

**Solutions**:

**PowerShell (Windows)**:
```powershell
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
Remove-Item -Recurse -Force node_modules, package-lock.json
npm install

# Check Node.js version
node --version
```

**Bash (Linux/Mac)**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node.js version
node --version
```

#### 6. ML Pipeline Errors
**Error**: `ML Pipeline not available` or `ImportError`

**Solutions**:
```bash
# Install ML dependencies
pip install -r requirements-ml.txt

# Check ML dependencies
python -c "import torch, torch_geometric, chromadb, neo4j; print('ML deps OK')"

# Check Neo4j connection
docker exec sabde-neo4j cypher-shell -u neo4j -p sabde_password "RETURN 1;"

# Check vector database
ls -la app/ml_pipeline/chroma_db_gene_mvp/
```

#### 7. ML Pipeline Initialization Failed
**Error**: `ML Pipeline initialization failed`

**Solutions**:
```bash
# Check ML pipeline status
curl http://localhost:8001/api/v1/research/ml-status

# Initialize ML pipeline manually
curl -X POST http://localhost:8001/api/v1/research/ml-initialize

# Check logs for specific errors
tail -f logs/app.log
```

#### 8. Real Data Mode Issues
**Error**: `entities=0` or `relationships=0` in API responses

**Solutions**:
```bash
# Check Neo4j connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'synthetica_password')); print('Neo4j OK' if driver.session().run('RETURN 1').single() else 'Neo4j Failed'); driver.close()"

# Check Neo4j data
python check_neo4j_stats.py

# Re-populate Neo4j if needed
python populate_neo4j_with_real_data.py

# Restart backend to pick up changes
# Kill existing backend process
taskkill /f /im python.exe
# Restart backend
python -m uvicorn app.main_ml:app --reload --host 0.0.0.0 --port 8001
```

**Error**: `Neo4j connection failed`

**Solutions**:
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Start Neo4j if not running
docker run -d --name synthetica-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/synthetica_password neo4j:latest

# Wait for Neo4j to start
Start-Sleep -Seconds 30

# Test connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'synthetica_password')); print('Connected!' if driver.session().run('RETURN 1').single() else 'Failed'); driver.close()"
```

### Debugging Steps

#### 1. Check Logs
```bash
# Backend logs
tail -f logs/app.log

# Database logs
docker logs sabde-postgres
docker logs sabde-neo4j
```

#### 2. Test Individual Components
```bash
# Test vector database
python -c "from app.services.vector_service import VectorService; print('Vector service OK')"

# Test knowledge graph
python -c "from app.services.knowledge_graph_service import KnowledgeGraphService; print('KG service OK')"

# Test API endpoints
curl -X GET http://localhost:8000/api/v1/health
```

#### 3. Verify Dependencies
```bash
# Check Python packages
pip list | grep -E "(langchain|fastapi|chromadb|neo4j)"

# Check Node.js packages
npm list --depth=0
```

## Production Deployment

### 1. Environment Configuration
```env
# Production settings
DEBUG=False
LOG_LEVEL=WARNING
SECRET_KEY=your_production_secret_key
DATABASE_URL=your_production_database_url
NEO4J_URI=your_production_neo4j_url
```

### 2. Database Setup
- Use managed database services (AWS RDS, Google Cloud SQL)
- Set up database backups and monitoring
- Configure connection pooling

### 3. Security Configuration
- Set up SSL/TLS certificates
- Configure firewall rules
- Enable authentication and authorization
- Set up monitoring and alerting

### 4. Scaling Considerations
- Use load balancers for multiple backend instances
- Set up database read replicas
- Configure CDN for static assets
- Implement caching strategies

## Maintenance

### Daily Tasks
- Monitor system health and performance
- Check error logs
- Verify backup processes

### Weekly Tasks
- Update dependencies
- Review security patches
- Analyze usage metrics

### Monthly Tasks
- Database maintenance
- Performance optimization
- Security audits

## Support

If you encounter issues not covered in this guide:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Verify all prerequisites are installed
4. Ensure all services are running
5. Check network connectivity
6. Review the architecture documentation

For additional help:
- Check the project README
- Review the API documentation at http://localhost:8000/docs
- Check the GitHub issues page
- Contact the development team

## Next Steps

Once the system is running:

1. **Explore the Interface**: Navigate through the different pages
2. **Test Research Queries**: Try various research questions
3. **Explore Knowledge Graph**: Search for entities and relationships
4. **Review Generated Hypotheses**: Analyze the AI-generated research ideas
5. **Customize Settings**: Adjust parameters and configurations
6. **Add Your Data**: Load your own scientific papers and datasets

### Real Data Mode Benefits

When running in Real Data Mode, you get:

- **15,183 Real Nodes**: Scientific abstracts, genes, and diseases from your data
- **10,025 Real Relationships**: Actual gene-disease associations from research
- **57 Unique Genes**: Including BRCA1, p53, EGFR, and other important genes
- **44 Disease Categories**: Cancer, stroke, diabetes, and other conditions
- **Evidence-Based Hypotheses**: Generated from real scientific literature
- **Graph Neural Network**: Using your trained GNN model for hypothesis generation

### Verification Commands

To verify your Real Data Mode is working:

```bash
# Check Neo4j statistics
python check_neo4j_stats.py

# Test API with real data
curl -X POST http://localhost:8001/api/v1/research/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "BRCA1 mutations and breast cancer", "domain": "biomedical", "max_hypotheses": 2}'

# Check if entities > 0 in response
```

The system is now ready for use in scientific research and hypothesis generation with your real biomedical data!
