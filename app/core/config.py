"""
Application configuration using Pydantic settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/sabde",
        description="PostgreSQL database URL"
    )
    
    # Neo4j Configuration
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="password", description="Neo4j password")
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./chroma_db",
        description="ChromaDB persistence directory"
    )
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="Pinecone API key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Pinecone environment")
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis URL")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-here", description="Secret key for JWT")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Token expiration time")
    
    # Model Configuration
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Default embedding model"
    )
    BIOBERT_MODEL: str = Field(
        default="dmis-lab/biobert-base-cased-v1.1",
        description="BioBERT model for biomedical NER"
    )
    
    # RAG Configuration
    MAX_RETRIEVAL_DOCS: int = Field(default=10, description="Maximum documents to retrieve")
    CHUNK_SIZE: int = Field(default=1000, description="Text chunk size for processing")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between text chunks")
    
    # ML Pipeline Configuration
    ML_MODE: bool = Field(default=False, description="Enable ML pipeline mode")
    ML_PIPELINE_ENABLED: bool = Field(default=False, description="ML pipeline enabled")
    GNN_MODEL_PATH: Optional[str] = Field(default=None, description="Path to GNN model")
    KNOWLEDGE_GRAPH_PATH: Optional[str] = Field(default=None, description="Path to knowledge graph")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
