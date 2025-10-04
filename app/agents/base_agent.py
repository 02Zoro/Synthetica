"""
Base agent class for all LangGraph agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import structlog
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core.config import settings

logger = structlog.get_logger()


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, model_type: str = "gpt-4o"):
        self.name = name
        self.model_type = model_type
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the language model based on model type."""
        if self.model_type.startswith("gpt"):
            return ChatOpenAI(
                model=self.model_type,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1
            )
        elif self.model_type.startswith("claude"):
            return ChatAnthropic(
                model=self.model_type,
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main logic."""
        pass
    
    def log_execution(self, state: Dict[str, Any], result: Dict[str, Any]):
        """Log agent execution for monitoring."""
        logger.info(
            f"Agent {self.name} executed",
            agent=self.name,
            input_keys=list(state.keys()),
            output_keys=list(result.keys())
        )
