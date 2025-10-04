"""
Streamlit version of Synthetica demo for easy deployment.
"""
import streamlit as st
import requests
import json
from typing import List, Dict, Any

# Page config
st.set_page_config(
    page_title="Synthetica Demo - AI Research Assistant",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Synthetica - AI Research Assistant")
st.subheader("Generate scientific research hypotheses with AI")

# Sidebar
st.sidebar.title("Demo Mode")
st.sidebar.info("This is a demo version with mock data. No API keys required!")

# Mock hypothesis generation function
def generate_mock_hypotheses(query: str, domain: str = "biomedical", max_hypotheses: int = 5) -> Dict[str, Any]:
    """Generate mock hypotheses based on query."""
    hypotheses = []
    
    if "cancer" in query.lower() or "tumor" in query.lower():
        hypotheses = [
            {
                "id": "hyp_001",
                "title": "Novel therapeutic target for cancer treatment",
                "description": "Based on the analysis of BRCA1 mutations and their role in breast cancer, we hypothesize that targeting specific DNA repair pathways could lead to more effective cancer treatments.",
                "rationale": "BRCA1 mutations disrupt DNA repair mechanisms, making cells more susceptible to DNA damage. Targeting these pathways could selectively kill cancer cells while sparing normal cells.",
                "testable_predictions": [
                    "Cells with BRCA1 mutations will be more sensitive to PARP inhibitors",
                    "Combination therapy with DNA damage agents will show synergistic effects",
                    "Patient survival rates will improve with targeted therapy"
                ],
                "methodology": [
                    "In vitro cell culture experiments",
                    "Xenograft mouse models",
                    "Clinical trial design with patient stratification"
                ],
                "expected_outcomes": [
                    "Identification of novel therapeutic targets",
                    "Improved patient response rates",
                    "Reduced side effects compared to current treatments"
                ],
                "confidence_score": 0.85
            }
        ]
    elif "diabetes" in query.lower() or "insulin" in query.lower():
        hypotheses = [
            {
                "id": "hyp_002",
                "title": "Metabolic reprogramming in diabetes pathogenesis",
                "description": "We hypothesize that dysregulation of mitochondrial function in pancreatic beta-cells leads to insulin resistance and type 2 diabetes development.",
                "rationale": "Insulin resistance is characterized by impaired glucose uptake and mitochondrial dysfunction. Targeting mitochondrial biogenesis could restore normal insulin sensitivity.",
                "testable_predictions": [
                    "Mitochondrial function will be impaired in diabetic patients",
                    "Mitochondrial biogenesis activators will improve insulin sensitivity",
                    "Metabolic flexibility will be restored with targeted interventions"
                ],
                "methodology": [
                    "Metabolomic profiling of patient samples",
                    "Mitochondrial function assays",
                    "Intervention studies with mitochondrial modulators"
                ],
                "expected_outcomes": [
                    "Identification of metabolic biomarkers",
                    "Development of targeted therapies",
                    "Improved diabetes management strategies"
                ],
                "confidence_score": 0.82
            }
        ]
    elif "alzheimer" in query.lower() or "dementia" in query.lower():
        hypotheses = [
            {
                "id": "hyp_003",
                "title": "Synaptic dysfunction in Alzheimer's disease progression",
                "description": "We hypothesize that the interaction between amyloid-beta and tau proteins disrupts synaptic function, leading to cognitive decline in Alzheimer's disease.",
                "rationale": "Amyloid-beta plaques and tau tangles are hallmarks of Alzheimer's disease. Their interaction may cause synaptic dysfunction and neuronal death.",
                "testable_predictions": [
                    "Synaptic density will be reduced in Alzheimer's patients",
                    "Amyloid-beta and tau co-localization will correlate with cognitive decline",
                    "Synaptic restoration will improve cognitive function"
                ],
                "methodology": [
                    "Post-mortem brain tissue analysis",
                    "Synaptic marker quantification",
                    "Cognitive assessment correlation studies"
                ],
                "expected_outcomes": [
                    "Identification of synaptic dysfunction mechanisms",
                    "Development of synaptic protection strategies",
                    "Improved diagnostic and therapeutic approaches"
                ],
                "confidence_score": 0.88
            }
        ]
    else:
        # Generic hypothesis
        hypotheses = [
            {
                "id": "hyp_generic",
                "title": "Cross-disciplinary research hypothesis",
                "description": f"Based on the analysis of the query '{query}', we hypothesize that interdisciplinary approaches combining multiple scientific domains could lead to novel insights and therapeutic strategies.",
                "rationale": "Complex biological systems require integrated approaches that span multiple disciplines and methodologies.",
                "testable_predictions": [
                    "Interdisciplinary collaboration will yield novel insights",
                    "Multi-omics approaches will reveal new biological mechanisms",
                    "Integrated therapeutic strategies will show improved efficacy"
                ],
                "methodology": [
                    "Literature review and meta-analysis",
                    "Cross-disciplinary collaboration frameworks",
                    "Integrated experimental design"
                ],
                "expected_outcomes": [
                    "Novel scientific insights",
                    "Improved research methodologies",
                    "Enhanced therapeutic development"
                ],
                "confidence_score": 0.75
            }
        ]
    
    # Limit to requested number
    hypotheses = hypotheses[:max_hypotheses]
    
    return {
        "query": query,
        "domain": domain,
        "hypotheses": hypotheses,
        "total_documents": 3,
        "processing_time": 1.5,
        "metadata": {
            "entities": 15,
            "relationships": 8,
            "iterations": 1,
            "status": "completed"
        }
    }

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔬 Research Query")
    
    # Input form
    with st.form("research_form"):
        query = st.text_area(
            "Enter your research question:",
            placeholder="e.g., What are the molecular mechanisms of breast cancer?",
            height=100
        )
        
        domain = st.selectbox(
            "Research Domain:",
            ["biomedical", "clinical", "pharmaceutical", "genetics"]
        )
        
        max_hypotheses = st.slider(
            "Maximum Hypotheses:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        submitted = st.form_submit_button("🚀 Generate Hypotheses", type="primary")

with col2:
    st.header("📊 Demo Info")
    st.info("**Demo Mode Active**")
    st.success("✅ No API keys required")
    st.warning("⚠️ Mock data only")
    
    st.header("🔗 API Endpoints")
    st.code("""
    POST /research/generate
    GET /health
    GET /docs
    """)

# Process the form
if submitted and query:
    with st.spinner("Generating hypotheses..."):
        # Generate mock hypotheses
        result = generate_mock_hypotheses(query, domain, max_hypotheses)
        
        # Display results
        st.header("🎯 Generated Hypotheses")
        
        for i, hypothesis in enumerate(result["hypotheses"], 1):
            with st.expander(f"Hypothesis {i}: {hypothesis['title']}", expanded=True):
                st.subheader(hypothesis["title"])
                st.write("**Description:**", hypothesis["description"])
                st.write("**Rationale:**", hypothesis["rationale"])
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Testable Predictions:**")
                    for prediction in hypothesis["testable_predictions"]:
                        st.write(f"• {prediction}")
                
                with col_b:
                    st.write("**Methodology:**")
                    for method in hypothesis["methodology"]:
                        st.write(f"• {method}")
                
                st.write("**Expected Outcomes:**")
                for outcome in hypothesis["expected_outcomes"]:
                    st.write(f"• {outcome}")
                
                # Confidence score
                confidence = hypothesis["confidence_score"]
                st.progress(confidence)
                st.write(f"**Confidence Score:** {confidence:.1%}")
        
        # Metadata
        st.header("📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", result["total_documents"])
        with col2:
            st.metric("Entities Found", result["metadata"]["entities"])
        with col3:
            st.metric("Relationships", result["metadata"]["relationships"])
        with col4:
            st.metric("Processing Time", f"{result['processing_time']:.1f}s")

elif submitted and not query:
    st.error("Please enter a research question!")

# Footer
st.markdown("---")
st.markdown("**Synthetica Demo** - AI-powered scientific research assistant")
st.markdown("Built with Streamlit for easy deployment and sharing")
