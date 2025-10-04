"""
Synthesis agent for generating hypotheses from extracted information.
"""
from typing import Dict, Any, List
import structlog
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from app.agents.base_agent import BaseAgent
from app.models.research import Hypothesis, Document, Entity, Relationship

logger = structlog.get_logger()


class SynthesisAgent(BaseAgent):
    """Agent responsible for synthesizing information into hypotheses."""
    
    def __init__(self):
        super().__init__("Synthesis_Agent", "gpt-4o")
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for hypothesis generation."""
        system_prompt = """You are a scientific research assistant specialized in generating novel, testable hypotheses from biomedical literature.

Your task is to analyze the provided scientific documents, entities, and relationships to generate innovative research hypotheses that could lead to new discoveries.

Guidelines:
1. Generate hypotheses that are novel and not obvious from the input data
2. Ensure hypotheses are testable and falsifiable
3. Provide clear rationale based on the evidence
4. Include specific, measurable predictions
5. Suggest appropriate methodologies for testing
6. Consider potential implications and applications

Focus on cross-domain connections and unexpected relationships that could lead to breakthrough discoveries."""

        human_prompt = """Based on the following information, generate {max_hypotheses} novel research hypotheses:

QUERY: {query}
DOMAIN: {domain}

DOCUMENTS:
{documents}

ENTITIES:
{entities}

RELATIONSHIPS:
{relationships}

Please generate hypotheses that:
- Are scientifically sound and testable
- Build upon the provided evidence
- Suggest novel research directions
- Include specific predictions and methodologies
- Consider potential cross-domain applications

Format each hypothesis with:
1. Title
2. Description
3. Rationale
4. Testable Predictions
5. Suggested Methodology
6. Expected Outcomes
7. Confidence Score (0-1)"""

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hypotheses from the provided information."""
        try:
            query = state.get("query", "")
            domain = state.get("domain", "biomedical")
            max_hypotheses = state.get("max_hypotheses", 5)
            documents = state.get("retrieved_documents", [])
            entities = state.get("entities", [])
            relationships = state.get("relationships", [])
            
            logger.info(f"Synthesis Agent generating {max_hypotheses} hypotheses")
            
            # Format input data for the prompt
            documents_text = self._format_documents(documents)
            entities_text = self._format_entities(entities)
            relationships_text = self._format_relationships(relationships)
            
            # Create the prompt
            prompt = self.prompt_template.format_messages(
                query=query,
                domain=domain,
                max_hypotheses=max_hypotheses,
                documents=documents_text,
                entities=entities_text,
                relationships=relationships_text
            )
            
            # Generate response
            response = await self.llm.ainvoke(prompt)
            hypotheses = self._parse_hypotheses(response.content, max_hypotheses)
            
            result = {
                "hypotheses": hypotheses,
                "total_hypotheses": len(hypotheses),
                "query": query,
                "domain": domain
            }
            
            self.log_execution(state, result)
            return result
            
        except Exception as e:
            logger.error(f"Synthesis Agent execution failed: {e}")
            return {
                "hypotheses": [],
                "total_hypotheses": 0,
                "error": str(e)
            }
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for the prompt."""
        if not documents:
            return "No documents provided"
        
        formatted = []
        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 documents
            formatted.append(f"{i}. {doc.title}\n   {doc.abstract[:500]}...")
        
        return "\n\n".join(formatted)
    
    def _format_entities(self, entities: List[Entity]) -> str:
        """Format entities for the prompt."""
        if not entities:
            return "No entities extracted"
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            if entity.label not in entity_groups:
                entity_groups[entity.label] = []
            entity_groups[entity.label].append(entity.text)
        
        formatted = []
        for label, texts in entity_groups.items():
            formatted.append(f"{label}: {', '.join(texts[:10])}")  # Limit to 10 per type
        
        return "\n".join(formatted)
    
    def _format_relationships(self, relationships: List[Relationship]) -> str:
        """Format relationships for the prompt."""
        if not relationships:
            return "No relationships extracted"
        
        formatted = []
        for rel in relationships[:10]:  # Limit to 10 relationships
            formatted.append(f"{rel.subject} --[{rel.predicate}]--> {rel.object}")
        
        return "\n".join(formatted)
    
    def _parse_hypotheses(self, response: str, max_hypotheses: int) -> List[Hypothesis]:
        """Parse the LLM response to extract hypotheses."""
        try:
            hypotheses = []
            lines = response.split('\n')
            current_hypothesis = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for hypothesis title
                if line.startswith('**') and line.endswith('**'):
                    if current_hypothesis:
                        hypotheses.append(current_hypothesis)
                    current_hypothesis = Hypothesis(
                        id=f"hyp_{len(hypotheses) + 1}",
                        title=line.strip('*'),
                        description="",
                        rationale="",
                        testable_predictions=[],
                        methodology=[],
                        expected_outcomes=[],
                        confidence_score=0.7
                    )
                elif current_hypothesis:
                    # Parse different sections
                    if line.startswith('Description:'):
                        current_hypothesis.description = line.replace('Description:', '').strip()
                    elif line.startswith('Rationale:'):
                        current_hypothesis.rationale = line.replace('Rationale:', '').strip()
                    elif line.startswith('Predictions:'):
                        predictions = line.replace('Predictions:', '').strip()
                        current_hypothesis.testable_predictions = [p.strip() for p in predictions.split(',')]
                    elif line.startswith('Methodology:'):
                        methodology = line.replace('Methodology:', '').strip()
                        current_hypothesis.methodology = [m.strip() for m in methodology.split(',')]
                    elif line.startswith('Outcomes:'):
                        outcomes = line.replace('Outcomes:', '').strip()
                        current_hypothesis.expected_outcomes = [o.strip() for o in outcomes.split(',')]
            
            if current_hypothesis:
                hypotheses.append(current_hypothesis)
            
            # Limit to requested number
            return hypotheses[:max_hypotheses]
            
        except Exception as e:
            logger.error(f"Failed to parse hypotheses: {e}")
            # Return a default hypothesis if parsing fails
            return [Hypothesis(
                id="hyp_1",
                title="Generated Hypothesis",
                description=response[:500],
                rationale="Based on the provided evidence",
                testable_predictions=["Prediction 1", "Prediction 2"],
                methodology=["Experimental validation", "Statistical analysis"],
                expected_outcomes=["Positive results", "Statistical significance"],
                confidence_score=0.6
            )]
