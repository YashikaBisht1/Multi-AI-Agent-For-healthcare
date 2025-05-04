# agents/__init__.py

from .summarize_tool import SummarizeTool
from .write_article_tool import WriteArticleTool
from .sanitize_data_tool import SanitizeDataTool
from .summarize_validator_agent import SummarizeValidatorAgent
from .write_article_validator_agent import WriteArticleValidatorAgent
from .sanitize_data_validator_agent import SanitizeValidatorAgent
from .refiner_agent import RefinerAgent # New import
from .validator_agent import ValidatorAgent  # New import
from .chatbot_agent import ChatbotAgent


class AgentManager:
    def __init__(self, max_retries=2, verbose=True):
        self.agents = {
            "summarize": SummarizeTool(max_retries=max_retries, verbose=verbose),
            "write_article": WriteArticleTool(max_retries=max_retries, verbose=verbose),
            "sanitize_data": SanitizeDataTool(max_retries=max_retries, verbose=verbose),
            "summarize_validator": SummarizeValidatorAgent(max_retries=max_retries, verbose=verbose),
            "write_article_validator": WriteArticleValidatorAgent(max_retries=max_retries, verbose=verbose),
            "sanitize_data_validator": SanitizeValidatorAgent(max_retries=max_retries, verbose=verbose),
            "refiner": RefinerAgent(max_retries=max_retries, verbose=verbose),      # New agent
            "validator": ValidatorAgent(max_retries=max_retries, verbose=verbose) , # New agent
            "chatbot": ChatbotAgent(max_retries=max_retries, verbose=verbose)       # New agent
        }

    def get_agent(self, agent_name, **kwargs):
        if agent_name == "chatbot":
            return ChatbotAgent(max_retries=2, verbose=True)  # Always use Ollama
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found.")
        return agent

