# SABDE Setup Guide

This guide will help you set up and run the State-of-the-Art Biomedical Discovery Engine (SABDE).

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose (optional)
- Git

## Quick Start

### Option 1: Local Development (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sabde
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Run the setup script**
   ```bash
   # On Windows
   scripts/start.bat
   
   # On Linux/Mac
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Option 2: Docker Compose

1. **Start all services**
   ```bash
   docker-compose up -d
   ```

2. **Initialize data**
   ```bash
   docker-compose exec sabde-backend python scripts/init_data.py
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Neo4j Browser: http://localhost:7474
   - PostgreSQL: localhost:5432

## Manual Setup

### Backend Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up databases**
   - Install PostgreSQL and create database
   - Install Neo4j and start service
   - Install Redis (optional, for caching)

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your database URLs and API keys
   ```

5. **Initialize data**
   ```bash
   python scripts/init_data.py
   ```

6. **Start the backend**
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the frontend**
   ```bash
   npm start
   ```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database URLs
DATABASE_URL=postgresql://user:password@localhost:5432/sabde
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db
PINECONE_API_KEY=your_pinecone_api_key  # Optional

# Redis
REDIS_URL=redis://localhost:6379

# Application
DEBUG=True
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

### Database Setup

#### PostgreSQL
```sql
CREATE DATABASE sabde;
CREATE USER sabde_user WITH PASSWORD 'sabde_password';
GRANT ALL PRIVILEGES ON DATABASE sabde TO sabde_user;
```

#### Neo4j
1. Download and install Neo4j Desktop or Community Edition
2. Start the service
3. Set password: `sabde_password`
4. Enable APOC plugin

#### Redis (Optional)
```bash
# Install Redis
# On Ubuntu/Debian
sudo apt-get install redis-server

# On macOS
brew install redis

# Start Redis
redis-server
```

## Monitoring Setup

### Prometheus and Grafana

1. **Start monitoring services**
   ```bash
   cd monitoring
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

2. **Access monitoring**
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3001 (admin/admin)

## Usage

### Basic Research Query

1. Open the frontend at http://localhost:3000
2. Navigate to the Research page
3. Enter your research question
4. Select domain and parameters
5. Click "Generate Hypotheses"
6. View the generated hypotheses and supporting evidence

### Knowledge Graph Exploration

1. Navigate to the Knowledge Graph page
2. Search for entities to find related concepts
3. Discover paths between different entities
4. Explore the interactive graph visualization

### API Usage

The backend provides a REST API for programmatic access:

```python
import requests

# Generate hypotheses
response = requests.post('http://localhost:8000/api/v1/research/generate', json={
    'query': 'What are the molecular mechanisms of Alzheimer\'s disease?',
    'domain': 'biomedical',
    'max_hypotheses': 5
})

hypotheses = response.json()['hypotheses']
```

## Troubleshooting

### Common Issues

1. **Port conflicts**
   - Change ports in docker-compose.yml or .env file
   - Kill existing processes using the ports

2. **Database connection errors**
   - Check database URLs in .env
   - Ensure databases are running
   - Verify credentials

3. **API key errors**
   - Ensure OpenAI and Anthropic API keys are set
   - Check API key validity and quotas

4. **Memory issues**
   - Increase Docker memory limits
   - Use smaller models for development

### Logs

Check logs for debugging:

```bash
# Backend logs
docker-compose logs sabde-backend

# All services
docker-compose logs
```

## Development

### Code Structure

```
sabde/
├── app/                    # FastAPI backend
│   ├── agents/            # LangGraph agents
│   ├── api/               # API routes
│   ├── core/              # Configuration
│   ├── models/            # Pydantic models
│   └── services/          # Business logic
├── frontend/              # React frontend
├── data/                  # Sample data
├── scripts/               # Setup scripts
├── monitoring/            # Monitoring config
└── tests/                 # Test files
```

### Adding New Features

1. **New Agent**: Add to `app/agents/`
2. **New API Endpoint**: Add to `app/api/routes/`
3. **New Frontend Page**: Add to `frontend/src/pages/`
4. **New Database Model**: Add to `app/models/`

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## Production Deployment

### Docker Production

1. **Build production images**
   ```bash
   docker-compose -f docker-compose.prod.yml build
   ```

2. **Deploy to production**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Cloud Deployment

- **AWS**: Use ECS, EKS, or Lambda
- **GCP**: Use Cloud Run or GKE
- **Azure**: Use Container Instances or AKS

### Environment Variables for Production

```env
DEBUG=False
LOG_LEVEL=WARNING
SECRET_KEY=your_production_secret_key
DATABASE_URL=your_production_database_url
NEO4J_URI=your_production_neo4j_url
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the logs
3. Create an issue on GitHub
4. Contact the development team

## License

This project is licensed under the MIT License - see the LICENSE file for details.
