"""
Demo agents that work without API keys for testing the system architecture.
"""
from typing import Dict, Any, List
import structlog
from app.models.research import Hypothesis, Document, Entity, Relationship
from datetime import datetime
import random

logger = structlog.get_logger()


class DemoRAGAgent:
    """Demo RAG agent that returns mock data."""
    
    def __init__(self):
        self.name = "Demo_RAG_Agent"
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock retrieved documents."""
        query = state.get("query", "")
        
        # Mock documents
        mock_documents = [
            Document(
                id="demo_001",
                title="BRCA1 mutations and breast cancer risk: a comprehensive analysis",
                abstract="Breast cancer is one of the most common malignancies affecting women worldwide. BRCA1 mutations have been identified as significant risk factors for hereditary breast and ovarian cancer syndrome.",
                authors=["Dr. Sarah Johnson", "Dr. Michael Chen"],
                journal="Nature Genetics",
                doi="10.1038/s41588-023-01456-7",
                keywords=["BRCA1", "breast cancer", "genetic mutations"],
                relevance_score=0.95
            ),
            Document(
                id="demo_002",
                title="Insulin resistance and type 2 diabetes: molecular mechanisms",
                abstract="Type 2 diabetes mellitus (T2DM) is characterized by insulin resistance and pancreatic beta-cell dysfunction. This comprehensive review examines the molecular mechanisms underlying insulin resistance.",
                authors=["Dr. Robert Kim", "Dr. Lisa Wang"],
                journal="Cell Metabolism",
                doi="10.1016/j.cmet.2023.07.012",
                keywords=["insulin resistance", "type 2 diabetes", "molecular mechanisms"],
                relevance_score=0.88
            ),
            Document(
                id="demo_003",
                title="Alzheimer's disease: amyloid-beta and tau protein interactions",
                abstract="Alzheimer's disease (AD) is the most common form of dementia, characterized by the accumulation of amyloid-beta plaques and neurofibrillary tangles composed of hyperphosphorylated tau protein.",
                authors=["Dr. Maria Garcia", "Dr. David Lee"],
                journal="Neuron",
                doi="10.1016/j.neuron.2023.08.015",
                keywords=["Alzheimer's disease", "amyloid-beta", "tau protein"],
                relevance_score=0.92
            )
        ]
        
        logger.info(f"Demo RAG Agent returning {len(mock_documents)} mock documents")
        
        return {
            "retrieved_documents": mock_documents,
            "total_documents": len(mock_documents),
            "query": query
        }


class DemoExtractionAgent:
    """Demo extraction agent that returns mock entities and relationships."""
    
    def __init__(self):
        self.name = "Demo_Extraction_Agent"
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock extracted entities and relationships."""
        
        # Mock entities
        mock_entities = [
            Entity(
                text="BRCA1",
                label="GENE",
                confidence=0.95,
                start_pos=0,
                end_pos=5
            ),
            Entity(
                text="breast cancer",
                label="DISEASE",
                confidence=0.90,
                start_pos=6,
                end_pos=18
            ),
            Entity(
                text="insulin resistance",
                label="DISEASE",
                confidence=0.88,
                start_pos=0,
                end_pos=17
            ),
            Entity(
                text="Alzheimer's disease",
                label="DISEASE",
                confidence=0.92,
                start_pos=0,
                end_pos=18
            ),
            Entity(
                text="amyloid-beta",
                label="PROTEIN",
                confidence=0.85,
                start_pos=19,
                end_pos=31
            )
        ]
        
        # Mock relationships
        mock_relationships = [
            Relationship(
                subject="BRCA1",
                predicate="ASSOCIATED_WITH",
                object="breast cancer",
                confidence=0.90,
                source_document="demo_001"
            ),
            Relationship(
                subject="insulin resistance",
                predicate="CAUSES",
                object="type 2 diabetes",
                confidence=0.88,
                source_document="demo_002"
            ),
            Relationship(
                subject="amyloid-beta",
                predicate="INTERACTS_WITH",
                object="tau protein",
                confidence=0.85,
                source_document="demo_003"
            )
        ]
        
        logger.info(f"Demo Extraction Agent returning {len(mock_entities)} entities and {len(mock_relationships)} relationships")
        
        return {
            "entities": mock_entities,
            "relationships": mock_relationships,
            "total_entities": len(mock_entities),
            "total_relationships": len(mock_relationships)
        }


class DemoSynthesisAgent:
    """Demo synthesis agent that returns mock hypotheses."""
    
    def __init__(self):
        self.name = "Demo_Synthesis_Agent"
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock generated hypotheses."""
        
        query = state.get("query", "")
        max_hypotheses = state.get("max_hypotheses", 3)
        
        # Mock hypotheses based on query
        mock_hypotheses = []
        
        if "cancer" in query.lower() or "tumor" in query.lower():
            mock_hypotheses = [
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
        elif "diabetes" in query.lower() or "insulin" in query.lower():
            mock_hypotheses = [
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
        elif "alzheimer" in query.lower() or "dementia" in query.lower():
            mock_hypotheses = [
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
            mock_hypotheses = [
                Hypothesis(
                    id="hyp_generic",
                    title="Cross-disciplinary research hypothesis",
                    description=f"Based on the analysis of the query '{query}', we hypothesize that interdisciplinary approaches combining multiple scientific domains could lead to novel insights and therapeutic strategies.",
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
        mock_hypotheses = mock_hypotheses[:max_hypotheses]
        
        logger.info(f"Demo Synthesis Agent returning {len(mock_hypotheses)} mock hypotheses")
        
        return {
            "hypotheses": mock_hypotheses,
            "total_hypotheses": len(mock_hypotheses),
            "query": query
        }


class DemoCriticAgent:
    """Demo critic agent that returns mock critiques."""
    
    def __init__(self):
        self.name = "Demo_Critic_Agent"
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock critiques."""
        
        hypotheses = state.get("hypotheses", [])
        critiques = []
        
        for hypothesis in hypotheses:
            # Mock critique
            critique = {
                "hypothesis_id": hypothesis.id,
                "status": "PASS",  # Demo always passes
                "feedback": f"Demo critique for {hypothesis.title}: The hypothesis is well-structured and testable. The methodology is appropriate and the expected outcomes are realistic.",
                "critique_text": f"""
**Overall Assessment: PASS**

**Strengths:**
- Clear and testable hypothesis
- Appropriate methodology
- Realistic expected outcomes
- Good scientific rationale

**Weaknesses:**
- Limited scope (demo mode)
- Needs more detailed experimental design
- Consider alternative explanations

**Recommendations:**
- Expand experimental design
- Include control groups
- Consider statistical power analysis
- Plan for potential confounding factors

**Methodological Concerns:**
- Ensure proper randomization
- Include appropriate controls
- Consider ethical implications

**Next Steps:**
- Develop detailed experimental protocol
- Secure funding and resources
- Establish collaborations
- Plan for data analysis
                """,
                "original_hypothesis": hypothesis
            }
            critiques.append(critique)
        
        logger.info(f"Demo Critic Agent returning {len(critiques)} mock critiques")
        
        return {
            "critiques": critiques,
            "overall_status": "PASS",
            "total_critiques": len(critiques)
        }
