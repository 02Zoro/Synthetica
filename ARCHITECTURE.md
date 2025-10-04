# SABDE Architecture Documentation

## Overview

The State-of-the-Art Biomedical Discovery Engine (SABDE) is a sophisticated AI-powered research assistant that generates novel scientific hypotheses from biomedical literature. This document explains the system architecture, design decisions, and implementation details.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SABDE System Architecture                │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React)          │  API Layer (FastAPI)              │
│  ┌─────────────────────┐   │  ┌─────────────────────────────┐   │
│  │ Research Interface  │   │  │ REST API Endpoints         │   │
│  │ Knowledge Graph UI  │   │  │ Authentication & Auth      │   │
│  │ Visualization       │   │  │ Request/Response Models    │   │
│  └─────────────────────┘   │  └─────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    Multi-Agent Orchestration (LangGraph)       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ RAG Agent   │ │ Extraction  │ │ Synthesis   │ │ Critic      ││
│  │ (Retrieval) │ │ Agent       │ │ Agent       │ │ Agent       ││
│  │             │ │ (BioBERT)   │ │ (GPT-4o)    │ │ (Claude)    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ Vector DB   │ │ Knowledge   │ │ Relational  │ │ Cache       ││
│  │ (ChromaDB)  │ │ Graph       │ │ DB          │ │ (Redis)     ││
│  │             │ │ (Neo4j)     │ │ (PostgreSQL)│ │             ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Multi-Agent System (LangGraph)

**Why LangGraph over LangChain?**
- **Stateful Orchestration**: LangGraph provides explicit state management across complex workflows
- **Conditional Logic**: Supports dynamic branching and self-correction loops
- **Human-in-the-Loop**: Enables scientist review and approval processes
- **Audit Trail**: Complete traceability of agent decisions and actions

**Agent Architecture:**
```python
# Each agent is a specialized component
class BaseAgent:
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.llm = self._initialize_llm()
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Agent-specific logic
        pass
```

**Workflow Design:**
```python
# LangGraph workflow with self-correction loop
workflow = StateGraph(ResearchState)
workflow.add_node("rag", self._rag_node)
workflow.add_node("extraction", self._extraction_node)
workflow.add_node("synthesis", self._synthesis_node)
workflow.add_node("critic", self._critic_node)
workflow.add_node("reflect", self._reflect_node)

# Conditional edges for self-correction
workflow.add_conditional_edges(
    "critic",
    self._should_continue,
    {"continue": "reflect", "end": END}
)
```

### 2. RAG (Retrieval-Augmented Generation) Pipeline

**Why ChromaDB over Pinecone/Weaviate?**
- **Local Development**: No external API dependencies
- **Cost Effective**: No per-query charges
- **Customization**: Full control over embedding models
- **Privacy**: Data stays on-premises

**Vector Search Implementation:**
```python
class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    async def search_documents(self, query: str, n_results: int = 10):
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
```

**Why Sentence Transformers?**
- **Domain Adaptation**: Can be fine-tuned for biomedical text
- **Efficiency**: Fast inference for real-time search
- **Quality**: State-of-the-art semantic understanding
- **Flexibility**: Easy to swap models (SciBERT, BioBERT)

### 3. Knowledge Graph (Neo4j)

**Why Neo4j over other graph databases?**
- **Cypher Query Language**: Intuitive graph querying
- **ACID Compliance**: Data integrity for scientific data
- **Graph Algorithms**: Built-in algorithms for path finding
- **Scalability**: Handles millions of nodes and relationships
- **APOC Plugin**: Advanced graph operations

**Graph Schema Design:**
```cypher
// Node constraints for data integrity
CREATE CONSTRAINT gene_id_unique FOR (g:Gene) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT protein_id_unique FOR (p:Protein) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT disease_id_unique FOR (d:Disease) REQUIRE d.id IS UNIQUE;

// Indexes for performance
CREATE INDEX gene_name_index FOR (g:Gene) ON (g.name);
CREATE INDEX protein_name_index FOR (p:Protein) ON (p.name);
```

**Entity Extraction Pipeline:**
```python
class ExtractionAgent:
    def __init__(self):
        # Initialize BioBERT for biomedical NER
        self.ner_pipeline = pipeline(
            "ner",
            model="dmis-lab/biobert-base-cased-v1.1",
            tokenizer=AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        )
    
    async def _extract_entities(self, text: str) -> List[Entity]:
        results = self.ner_pipeline(text)
        entities = []
        for result in results:
            entity = Entity(
                text=result["word"],
                label=result["entity_group"],
                confidence=result["score"]
            )
            entities.append(entity)
        return entities
```

### 4. FastAPI Microservices

**Why FastAPI over Flask/Django?**
- **Performance**: Async support for high concurrency
- **Type Safety**: Pydantic models with automatic validation
- **Documentation**: Auto-generated OpenAPI/Swagger docs
- **Dependency Injection**: Clean separation of concerns
- **Modern Python**: Uses Python 3.6+ features

**API Design Patterns:**
```python
# Dependency injection for services
@router.post("/research/generate")
async def generate_hypotheses(
    query: ResearchQuery,
    vector_service: VectorService = Depends(),
    kg_service: KnowledgeGraphService = Depends()
):
    # Business logic with injected dependencies
    pass

# Pydantic models for validation
class ResearchQuery(BaseModel):
    query: str = Field(..., description="Research question")
    domain: ResearchDomain = Field(default=ResearchDomain.BIOMEDICAL)
    max_hypotheses: int = Field(default=5, ge=1, le=20)
```

### 5. React Frontend

**Why React over Vue/Angular?**
- **Ecosystem**: Rich library ecosystem (Cytoscape.js, D3.js)
- **Component Reusability**: Modular UI components
- **State Management**: Easy integration with complex state
- **Performance**: Virtual DOM for efficient updates
- **Developer Experience**: Hot reloading, debugging tools

**Component Architecture:**
```javascript
// Page-level components
const ResearchPage = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  
  const handleSubmit = async (values) => {
    setLoading(true);
    const response = await axios.post('/api/v1/research/generate', values);
    setResults(response.data);
    setLoading(false);
  };
  
  return (
    <div>
      <ResearchForm onSubmit={handleSubmit} />
      {loading && <LoadingSpinner />}
      {results && <ResultsDisplay results={results} />}
    </div>
  );
};
```

### 6. Database Layer

**PostgreSQL for Metadata:**
- **ACID Compliance**: Ensures data consistency
- **JSON Support**: Flexible schema for metadata
- **Full-Text Search**: Built-in search capabilities
- **Mature Ecosystem**: Extensive tooling and support

**Redis for Caching:**
- **Performance**: Sub-millisecond response times
- **Scalability**: Horizontal scaling support
- **Data Structures**: Rich data types for complex caching
- **Persistence**: Optional data persistence

### 7. Monitoring and Observability

**Why Prometheus + Grafana?**
- **Industry Standard**: Widely adopted monitoring stack
- **Flexibility**: Custom metrics and dashboards
- **Integration**: Works with all major cloud providers
- **Cost Effective**: Open-source solution

**Custom Metrics:**
```python
# LLM usage tracking
llm_tokens_total = Counter('llm_tokens_total', 'Total LLM tokens used', ['model', 'type'])

# Agent performance
agent_execution_duration = Histogram('agent_execution_duration_seconds', 'Agent execution time', ['agent_name'])

# Vector database operations
vector_db_operations = Counter('vector_db_operations_total', 'Vector DB operations', ['operation'])
```

## Data Flow

### 1. Research Query Processing

```
User Query → API Validation → RAG Agent → Vector Search → Document Retrieval
```

**Step-by-step:**
1. User submits research question via React frontend
2. FastAPI validates input using Pydantic models
3. RAG Agent processes query and searches vector database
4. Relevant documents are retrieved and ranked by relevance

### 2. Entity and Relationship Extraction

```
Retrieved Documents → BioBERT NER → Entity Extraction → Relationship Mapping → Neo4j Storage
```

**Step-by-step:**
1. Retrieved documents are processed by Extraction Agent
2. BioBERT identifies biomedical entities (genes, proteins, diseases)
3. Relationships between entities are extracted
4. Entities and relationships are stored in Neo4j knowledge graph

### 3. Hypothesis Generation

```
Extracted Knowledge → Synthesis Agent → LLM Processing → Hypothesis Generation → Critic Validation
```

**Step-by-step:**
1. Synthesis Agent combines retrieved documents and extracted knowledge
2. GPT-4o generates novel hypotheses based on the combined information
3. Critic Agent (Claude) validates hypotheses for scientific rigor
4. Self-correction loop refines hypotheses if needed

### 4. Self-Correction Loop

```
Hypothesis → Critic Agent → Validation → [PASS/FAIL/REVISION] → [Reflect → Regenerate]
```

**Why Self-Correction?**
- **Quality Assurance**: Ensures scientific rigor
- **Error Reduction**: Catches logical inconsistencies
- **Iterative Improvement**: Refines hypotheses based on feedback
- **Transparency**: Provides audit trail of corrections

## Design Decisions

### 1. Why Multi-Agent Architecture?

**Single Agent Limitations:**
- **Context Window**: Limited by LLM context limits
- **Specialization**: Cannot excel at all tasks simultaneously
- **Scalability**: Difficult to scale individual components
- **Debugging**: Hard to isolate issues in monolithic systems

**Multi-Agent Benefits:**
- **Specialization**: Each agent excels at specific tasks
- **Modularity**: Easy to modify or replace individual agents
- **Scalability**: Independent scaling of components
- **Debugging**: Clear separation of concerns

### 2. Why LangGraph over LangChain?

**LangChain Limitations:**
- **Linear Workflows**: Difficult to model complex branching
- **State Management**: Limited state persistence
- **Error Handling**: Basic error recovery mechanisms
- **Human Interaction**: Limited HITL support

**LangGraph Advantages:**
- **Graph-Based**: Natural modeling of complex workflows
- **Stateful**: Explicit state management across agents
- **Conditional Logic**: Dynamic branching based on results
- **Human-in-the-Loop**: Built-in HITL capabilities

### 3. Why BioBERT for NER?

**General Models (BERT, RoBERTa):**
- **Domain Mismatch**: Trained on general text, not biomedical
- **Vocabulary**: Limited biomedical terminology
- **Performance**: Lower accuracy on biomedical tasks

**BioBERT Advantages:**
- **Domain Pre-training**: Trained on PubMed abstracts and PMC articles
- **Biomedical Vocabulary**: Extensive medical terminology
- **Performance**: State-of-the-art on biomedical NER tasks
- **Fine-tuning**: Easy to adapt to specific tasks

### 4. Why Neo4j for Knowledge Graph?

**Relational Databases:**
- **Graph Queries**: Complex joins for graph traversal
- **Performance**: Slow for deep graph queries
- **Scalability**: Limited by join complexity

**Neo4j Advantages:**
- **Native Graph**: Optimized for graph operations
- **Cypher**: Intuitive graph query language
- **Performance**: Fast graph traversal algorithms
- **Algorithms**: Built-in graph algorithms (PageRank, community detection)

### 5. Why FastAPI over Flask?

**Flask Limitations:**
- **Synchronous**: Limited async support
- **Validation**: Manual request validation
- **Documentation**: Manual API documentation
- **Type Safety**: Limited type checking

**FastAPI Advantages:**
- **Async Support**: Native async/await support
- **Automatic Validation**: Pydantic model validation
- **Auto Documentation**: OpenAPI/Swagger generation
- **Type Safety**: Full type checking with mypy

## Security Considerations

### 1. API Security
- **Authentication**: JWT tokens for user sessions
- **Authorization**: Role-based access control
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Pydantic model validation

### 2. Data Privacy
- **Local Processing**: Sensitive data stays on-premises
- **Encryption**: Data encryption at rest and in transit
- **Access Control**: Database-level access controls
- **Audit Logging**: Complete audit trail

### 3. Model Security
- **API Keys**: Secure storage of LLM API keys
- **Request Logging**: Monitor for suspicious activity
- **Output Filtering**: Sanitize generated content
- **Rate Limiting**: Prevent API quota exhaustion

## Performance Optimizations

### 1. Caching Strategy
- **Redis**: Cache frequent queries and results
- **Vector Cache**: Cache embedding computations
- **Graph Cache**: Cache graph traversal results
- **LLM Cache**: Cache similar LLM responses

### 2. Database Optimization
- **Indexing**: Strategic database indexes
- **Connection Pooling**: Efficient database connections
- **Query Optimization**: Optimized Cypher queries
- **Batch Operations**: Bulk data operations

### 3. Async Processing
- **Background Tasks**: Long-running operations
- **Queue System**: Task queuing for scalability
- **Parallel Processing**: Concurrent agent execution
- **Streaming**: Real-time result streaming

## Scalability Considerations

### 1. Horizontal Scaling
- **Microservices**: Independent service scaling
- **Load Balancing**: Distribute traffic across instances
- **Database Sharding**: Partition data across databases
- **CDN**: Content delivery for static assets

### 2. Vertical Scaling
- **Resource Monitoring**: Track CPU, memory, disk usage
- **Auto-scaling**: Automatic resource adjustment
- **Performance Tuning**: Optimize for specific workloads
- **Capacity Planning**: Plan for growth

### 3. Cloud Deployment
- **Container Orchestration**: Kubernetes for container management
- **Service Mesh**: Istio for service communication
- **Monitoring**: Cloud-native monitoring solutions
- **CI/CD**: Automated deployment pipelines

## Future Enhancements

### 1. Advanced AI Features
- **Graph Neural Networks**: Deep learning on knowledge graphs
- **Reinforcement Learning**: Adaptive hypothesis generation
- **Multi-modal AI**: Process images, tables, and text
- **Federated Learning**: Collaborative model training

### 2. Privacy and Security
- **Homomorphic Encryption**: Compute on encrypted data
- **Differential Privacy**: Privacy-preserving analytics
- **Secure Multi-party Computation**: Collaborative research
- **Blockchain**: Immutable research records

### 3. User Experience
- **Natural Language Interface**: Conversational research assistant
- **Visual Analytics**: Interactive data visualization
- **Collaborative Features**: Multi-user research sessions
- **Mobile Support**: Mobile-optimized interface

## Conclusion

The SABDE architecture represents a state-of-the-art approach to AI-powered scientific research. By combining multi-agent systems, knowledge graphs, and advanced NLP techniques, it provides a powerful platform for generating novel research hypotheses while maintaining scientific rigor and transparency.

The modular design allows for easy extension and modification, while the comprehensive monitoring and observability features ensure reliable operation in production environments. The system is designed to scale from individual researchers to large research institutions, providing a foundation for the future of AI-assisted scientific discovery.
