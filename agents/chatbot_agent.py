
from langchain_community.llms import Ollama
from .agent_base import AgentBase
# agents/chatbot_agent.py

class ChatbotAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__("ChatbotAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, user_input):
        """
        Process user input and return AI-generated response.
        """
        messages = [
            {"role": "system", "content": "You are a helpful and knowledgeable AI assistant specialized in medical research and healthcare."},
            {"role": "user", "content": user_input}
        ]

        response = self.call_llama(messages)
        return response
