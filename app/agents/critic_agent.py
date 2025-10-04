"""
Critic agent for validating and improving generated hypotheses.
"""
from typing import Dict, Any, List
import structlog
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from app.agents.base_agent import BaseAgent
from app.models.research import Hypothesis

logger = structlog.get_logger()


class CriticAgent(BaseAgent):
    """Agent responsible for critiquing and validating hypotheses."""
    
    def __init__(self):
        super().__init__("Critic_Agent", "claude-3-sonnet-20240229")
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for hypothesis critique."""
        system_prompt = """You are a rigorous scientific critic and peer reviewer with expertise in biomedical research methodology and experimental design.

Your task is to evaluate research hypotheses for:
1. Scientific validity and soundness
2. Testability and falsifiability
3. Methodological rigor
4. Logical consistency
5. Potential biases or limitations
6. Feasibility of testing
7. Novelty and significance

Provide constructive feedback and suggest improvements where needed. Be thorough but fair in your assessment."""

        human_prompt = """Please critically evaluate the following research hypothesis:

HYPOTHESIS TITLE: {title}
DESCRIPTION: {description}
RATIONALE: {rationale}
PREDICTIONS: {predictions}
METHODOLOGY: {methodology}
EXPECTED OUTCOMES: {outcomes}
CONFIDENCE: {confidence}

Please provide:
1. Overall assessment (PASS/FAIL/NEEDS_REVISION)
2. Strengths of the hypothesis
3. Weaknesses and limitations
4. Specific suggestions for improvement
5. Methodological concerns
6. Potential alternative explanations
7. Recommended next steps

Be specific and constructive in your feedback."""

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Critique the generated hypotheses."""
        try:
            hypotheses = state.get("hypotheses", [])
            query = state.get("query", "")
            
            logger.info(f"Critic Agent evaluating {len(hypotheses)} hypotheses")
            
            critiques = []
            overall_status = "PASS"
            
            for hypothesis in hypotheses:
                critique = await self._critique_hypothesis(hypothesis)
                critiques.append(critique)
                
                # Update overall status based on individual critiques
                if critique["status"] == "FAIL":
                    overall_status = "FAIL"
                elif critique["status"] == "NEEDS_REVISION" and overall_status != "FAIL":
                    overall_status = "NEEDS_REVISION"
            
            result = {
                "critiques": critiques,
                "overall_status": overall_status,
                "total_critiques": len(critiques),
                "query": query
            }
            
            self.log_execution(state, result)
            return result
            
        except Exception as e:
            logger.error(f"Critic Agent execution failed: {e}")
            return {
                "critiques": [],
                "overall_status": "FAIL",
                "total_critiques": 0,
                "error": str(e)
            }
    
    async def _critique_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Critique a single hypothesis."""
        try:
            # Format the hypothesis for critique
            prompt = self.prompt_template.format_messages(
                title=hypothesis.title,
                description=hypothesis.description,
                rationale=hypothesis.rationale,
                predictions=", ".join(hypothesis.testable_predictions),
                methodology=", ".join(hypothesis.methodology),
                outcomes=", ".join(hypothesis.expected_outcomes),
                confidence=hypothesis.confidence_score
            )
            
            # Get critique from Claude
            response = await self.llm.ainvoke(prompt)
            critique_text = response.content
            
            # Parse the critique to extract status and feedback
            status, feedback = self._parse_critique(critique_text)
            
            return {
                "hypothesis_id": hypothesis.id,
                "status": status,
                "feedback": feedback,
                "critique_text": critique_text,
                "original_hypothesis": hypothesis
            }
            
        except Exception as e:
            logger.error(f"Failed to critique hypothesis {hypothesis.id}: {e}")
            return {
                "hypothesis_id": hypothesis.id,
                "status": "FAIL",
                "feedback": f"Critique failed: {str(e)}",
                "critique_text": "",
                "original_hypothesis": hypothesis
            }
    
    def _parse_critique(self, critique_text: str) -> tuple[str, str]:
        """Parse the critique text to extract status and feedback."""
        try:
            lines = critique_text.split('\n')
            status = "PASS"  # Default status
            feedback = critique_text
            
            for line in lines:
                line = line.strip().upper()
                if "OVERALL ASSESSMENT:" in line or "ASSESSMENT:" in line:
                    if "FAIL" in line:
                        status = "FAIL"
                    elif "NEEDS_REVISION" in line or "REVISION" in line:
                        status = "NEEDS_REVISION"
                    elif "PASS" in line:
                        status = "PASS"
                elif "STATUS:" in line:
                    if "FAIL" in line:
                        status = "FAIL"
                    elif "NEEDS_REVISION" in line or "REVISION" in line:
                        status = "NEEDS_REVISION"
                    elif "PASS" in line:
                        status = "PASS"
            
            return status, feedback
            
        except Exception as e:
            logger.error(f"Failed to parse critique: {e}")
            return "FAIL", f"Failed to parse critique: {str(e)}"
