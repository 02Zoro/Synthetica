"""
Pydantic models for research-related API requests and responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ResearchDomain(str, Enum):
    """Supported research domains."""
    BIOMEDICAL = "biomedical"
    CLINICAL = "clinical"
    PHARMACEUTICAL = "pharmaceutical"
    GENETICS = "genetics"
    NEUROSCIENCE = "neuroscience"
    ONCOLOGY = "oncology"


class HypothesisStatus(str, Enum):
    """Hypothesis validation status."""
    DRAFT = "draft"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ResearchQuery(BaseModel):
    """Research query request model."""
    query: str = Field(..., description="Research question or topic")
    domain: ResearchDomain = Field(default=ResearchDomain.BIOMEDICAL, description="Research domain")
    max_hypotheses: int = Field(default=5, ge=1, le=20, description="Maximum number of hypotheses to generate")
    include_sources: bool = Field(default=True, description="Include source documents")
    depth: int = Field(default=2, ge=1, le=5, description="Research depth level")


class Document(BaseModel):
    """Scientific document model."""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    abstract: str = Field(..., description="Document abstract")
    authors: List[str] = Field(default=[], description="Document authors")
    publication_date: Optional[datetime] = Field(default=None, description="Publication date")
    journal: Optional[str] = Field(default=None, description="Journal name")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    url: Optional[str] = Field(default=None, description="Document URL")
    keywords: List[str] = Field(default=[], description="Document keywords")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class Entity(BaseModel):
    """Named entity extracted from text."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label/type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    start_pos: int = Field(..., description="Start position in text")
    end_pos: int = Field(..., description="End position in text")


class Relationship(BaseModel):
    """Relationship between entities."""
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relationship type")
    object: str = Field(..., description="Object entity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relationship confidence")
    source_document: str = Field(..., description="Source document ID")


class Hypothesis(BaseModel):
    """Generated research hypothesis."""
    id: str = Field(..., description="Hypothesis ID")
    title: str = Field(..., description="Hypothesis title")
    description: str = Field(..., description="Detailed hypothesis description")
    rationale: str = Field(..., description="Scientific rationale")
    testable_predictions: List[str] = Field(default=[], description="Testable predictions")
    methodology: List[str] = Field(default=[], description="Suggested methodology")
    expected_outcomes: List[str] = Field(default=[], description="Expected outcomes")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    status: HypothesisStatus = Field(default=HypothesisStatus.DRAFT, description="Validation status")
    supporting_evidence: List[Document] = Field(default=[], description="Supporting documents")
    entities: List[Entity] = Field(default=[], description="Extracted entities")
    relationships: List[Relationship] = Field(default=[], description="Extracted relationships")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class ResearchResponse(BaseModel):
    """Research response model."""
    query: str = Field(..., description="Original query")
    domain: ResearchDomain = Field(..., description="Research domain")
    hypotheses: List[Hypothesis] = Field(..., description="Generated hypotheses")
    total_documents: int = Field(..., description="Total documents processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class AgentStatus(BaseModel):
    """Agent execution status."""
    agent_name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Execution status")
    start_time: datetime = Field(..., description="Start time")
    end_time: Optional[datetime] = Field(default=None, description="End time")
    duration: Optional[float] = Field(default=None, description="Duration in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class WorkflowStatus(BaseModel):
    """Workflow execution status."""
    workflow_id: str = Field(..., description="Workflow ID")
    status: str = Field(..., description="Overall status")
    agents: List[AgentStatus] = Field(..., description="Agent statuses")
    current_step: str = Field(..., description="Current execution step")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress percentage")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
