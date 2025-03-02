# agents/summarize_agent.py

from .agent_base import AgentBase


class SummarizeTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SummarizeTool", max_retries=max_retries, verbose=verbose)
        self.temperature = 0.7  # Initial model randomness
        self.max_tokens = 300  # Initial summary length

    def execute(self, text):
        """
        Generates a summary of the given medical text.
        """
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that summarizes medical texts concisely and accurately."},
            {"role": "user", "content": f"Summarize the following medical text concisely:\n\n{text}\n\nSummary:"}
        ]

        summary = self.call_llama(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return summary
