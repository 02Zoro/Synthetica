"""
Simple demo server for SABDE without complex configuration.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Pydantic models
class ResearchQuery(BaseModel):
    query: str
    domain: str = "biomedical"
    max_hypotheses: int = 5

class Hypothesis(BaseModel):
    id: str
    title: str
    description: str
    rationale: str
    testable_predictions: List[str]
    methodology: List[str]
    expected_outcomes: List[str]
    confidence_score: float

class ResearchResponse(BaseModel):
    query: str
    domain: str
    hypotheses: List[Hypothesis]
    total_documents: int
    processing_time: float
    metadata: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="Synthetica Demo - Scientific Research Assistant",
    description="AI-powered hypothesis generation (Demo Mode)",
    version="1.0.0-demo"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Synthetica Demo - Scientific Research Assistant",
        "mode": "demo",
        "description": "AI-powered hypothesis generation",
        "endpoints": {
            "health": "/health",
            "research": "/research/generate",
            "docs": "/docs"
        }
    }

@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "synthetica-demo"}

@app.post("/research/generate", response_model=ResearchResponse)
@app.post("/api/v1/research/generate", response_model=ResearchResponse)
async def generate_hypotheses(query: ResearchQuery):
    """Generate research hypotheses from a query."""
    try:
        logger.info(f"Generating hypotheses for query: {query.query}")
        
        # Mock hypotheses based on query
        hypotheses = []
        
        if "cancer" in query.query.lower() or "tumor" in query.query.lower():
            hypotheses = [
                Hypothesis(
                    id="hyp_001",
                    title="Novel therapeutic target for cancer treatment",
                    description="Based on the analysis of BRCA1 mutations and their role in breast cancer, we hypothesize that targeting specific DNA repair pathways could lead to more effective cancer treatments.",
                    rationale="BRCA1 mutations disrupt DNA repair mechanisms, making cells more susceptible to DNA damage. Targeting these pathways could selectively kill cancer cells while sparing normal cells.",
                    testable_predictions=[
                        "Cells with BRCA1 mutations will be more sensitive to PARP inhibitors",
                        "Combination therapy with DNA damage agents will show synergistic effects",
                        "Patient survival rates will improve with targeted therapy"
                    ],
                    methodology=[
                        "In vitro cell culture experiments",
                        "Xenograft mouse models",
                        "Clinical trial design with patient stratification"
                    ],
                    expected_outcomes=[
                        "Identification of novel therapeutic targets",
                        "Improved patient response rates",
                        "Reduced side effects compared to current treatments"
                    ],
                    confidence_score=0.85
                )
            ]
        elif "diabetes" in query.query.lower() or "insulin" in query.query.lower():
            hypotheses = [
                Hypothesis(
                    id="hyp_002",
                    title="Metabolic reprogramming in diabetes pathogenesis",
                    description="We hypothesize that dysregulation of mitochondrial function in pancreatic beta-cells leads to insulin resistance and type 2 diabetes development.",
                    rationale="Insulin resistance is characterized by impaired glucose uptake and mitochondrial dysfunction. Targeting mitochondrial biogenesis could restore normal insulin sensitivity.",
                    testable_predictions=[
                        "Mitochondrial function will be impaired in diabetic patients",
                        "Mitochondrial biogenesis activators will improve insulin sensitivity",
                        "Metabolic flexibility will be restored with targeted interventions"
                    ],
                    methodology=[
                        "Metabolomic profiling of patient samples",
                        "Mitochondrial function assays",
                        "Intervention studies with mitochondrial modulators"
                    ],
                    expected_outcomes=[
                        "Identification of metabolic biomarkers",
                        "Development of targeted therapies",
                        "Improved diabetes management strategies"
                    ],
                    confidence_score=0.82
                )
            ]
        elif "alzheimer" in query.query.lower() or "dementia" in query.query.lower():
            hypotheses = [
                Hypothesis(
                    id="hyp_003",
                    title="Synaptic dysfunction in Alzheimer's disease progression",
                    description="We hypothesize that the interaction between amyloid-beta and tau proteins disrupts synaptic function, leading to cognitive decline in Alzheimer's disease.",
                    rationale="Amyloid-beta plaques and tau tangles are hallmarks of Alzheimer's disease. Their interaction may cause synaptic dysfunction and neuronal death.",
                    testable_predictions=[
                        "Synaptic density will be reduced in Alzheimer's patients",
                        "Amyloid-beta and tau co-localization will correlate with cognitive decline",
                        "Synaptic restoration will improve cognitive function"
                    ],
                    methodology=[
                        "Post-mortem brain tissue analysis",
                        "Synaptic marker quantification",
                        "Cognitive assessment correlation studies"
                    ],
                    expected_outcomes=[
                        "Identification of synaptic dysfunction mechanisms",
                        "Development of synaptic protection strategies",
                        "Improved diagnostic and therapeutic approaches"
                    ],
                    confidence_score=0.88
                )
            ]
        else:
            # Generic hypothesis
            hypotheses = [
                Hypothesis(
                    id="hyp_generic",
                    title="Cross-disciplinary research hypothesis",
                    description=f"Based on the analysis of the query '{query.query}', we hypothesize that interdisciplinary approaches combining multiple scientific domains could lead to novel insights and therapeutic strategies.",
                    rationale="Complex biological systems require integrated approaches that span multiple disciplines and methodologies.",
                    testable_predictions=[
                        "Interdisciplinary collaboration will yield novel insights",
                        "Multi-omics approaches will reveal new biological mechanisms",
                        "Integrated therapeutic strategies will show improved efficacy"
                    ],
                    methodology=[
                        "Literature review and meta-analysis",
                        "Cross-disciplinary collaboration frameworks",
                        "Integrated experimental design"
                    ],
                    expected_outcomes=[
                        "Novel scientific insights",
                        "Improved research methodologies",
                        "Enhanced therapeutic development"
                    ],
                    confidence_score=0.75
                )
            ]
        
        # Limit to requested number
        hypotheses = hypotheses[:query.max_hypotheses]
        
        response = ResearchResponse(
            query=query.query,
            domain=query.domain,
            hypotheses=hypotheses,
            total_documents=3,  # Mock document count
            processing_time=1.5,  # Mock processing time
            metadata={
                "entities": 15,
                "relationships": 8,
                "iterations": 1,
                "status": "completed"
            }
        )
        
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        return response
        
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research/entities/{entity_name}")
@app.get("/api/v1/research/entities/{entity_name}")
async def find_related_entities(entity_name: str):
    """Find entities related to a given entity."""
    # Mock related entities
    mock_entities = [
        {"name": "BRCA1", "type": "Gene", "distance": 1},
        {"name": "breast cancer", "type": "Disease", "distance": 1},
        {"name": "PARP inhibitors", "type": "Drug", "distance": 2},
        {"name": "DNA repair", "type": "Pathway", "distance": 2}
    ]
    
    return {
        "entity_name": entity_name,
        "related_entities": mock_entities,
        "total_found": len(mock_entities)
    }

@app.get("/research/paths/{entity1}/{entity2}")
@app.get("/api/v1/research/paths/{entity1}/{entity2}")
async def find_entity_paths(entity1: str, entity2: str):
    """Find paths between two entities."""
    # Mock paths
    mock_paths = [
        {
            "nodes": [
                {"name": entity1, "type": "Gene"},
                {"name": "protein interaction", "type": "Relationship"},
                {"name": entity2, "type": "Disease"}
            ],
            "relationships": [
                {"type": "ASSOCIATED_WITH", "confidence": 0.85}
            ],
            "length": 2
        }
    ]
    
    return {
        "entity1": entity1,
        "entity2": entity2,
        "paths": mock_paths,
        "total_paths": len(mock_paths)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.simple_demo:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
