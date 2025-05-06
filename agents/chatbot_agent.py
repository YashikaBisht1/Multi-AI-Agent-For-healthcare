
from .agent_base import AgentBase

class ChatbotAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__("ChatbotAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, user_input):
        # Always use Ollama (call_llama)
        messages = [
            {"role": "system", "content": (
                "You are a highly knowledgeable, careful, and ethical medical assistant. "
                "Always provide evidence-based, up-to-date, and safe advice. "
                "If you are unsure, say so and recommend consulting a healthcare professional. "
                "Cite guidelines or reputable sources when possible."
            )},
            {"role": "user", "content": user_input}
        ]
        return self.call_llama(messages)

