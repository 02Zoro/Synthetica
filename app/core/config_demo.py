"""
Demo configuration for running SABDE without API keys.
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class DemoSettings(BaseSettings):
    """Demo settings for running without API keys."""
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=True, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Demo Mode - No API keys required
    DEMO_MODE: bool = Field(default=True, description="Run in demo mode")
    OPENAI_API_KEY: Optional[str] = Field(default="demo-key", description="Demo OpenAI key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default="demo-key", description="Demo Anthropic key")
    
    # Database Configuration (use in-memory for demo)
    DATABASE_URL: str = Field(
        default="sqlite:///./demo.db",
        description="SQLite database for demo"
    )
    
    # Neo4j Configuration (optional for demo)
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="demo", description="Neo4j password")
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./chroma_db",
        description="ChromaDB persistence directory"
    )
    
    # Redis Configuration (optional for demo)
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis URL")
    
    # Security
    SECRET_KEY: str = Field(default="demo-secret-key", description="Demo secret key")
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
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global demo settings instance
demo_settings = DemoSettings()
