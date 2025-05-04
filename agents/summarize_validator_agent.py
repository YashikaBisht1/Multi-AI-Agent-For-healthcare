# agents/summarize_validator_agent.py
from .agent_base import AgentBase
import numpy as np

class SummarizeValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SummarizeValidatorAgent", max_retries=max_retries, verbose=verbose)
        self.validation_history = []  # Store validation feedback
        self.temperature = 0.7
        self.max_tokens = 512

    def execute(self, original_text, summary, human_rating=None):
        """
        Validates the accuracy and conciseness of a medical summary.
        """
        system_message = "You are an AI assistant that validates summaries of medical texts."
        user_content = (
            "Given the original text and its summary, assess whether the summary accurately and concisely captures the key points.\n"
            "Provide a brief analysis and rate the summary on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
            f"Original Text:\n{original_text}\n\n"
            f"Summary:\n{summary}\n\n"
            "Validation Report:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        validation_response = self.call_llama(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        ai_rating = self.extract_validation_score(validation_response)

        # Use provided human_rating or default to 3 if not given
        if human_rating is None:
            human_rating = 3

        average_score = (ai_rating + human_rating) / 2

        self.optimize_with_rl()

        return validation_response, ai_rating, human_rating, average_score

    def extract_validation_score(self, response):
        """
        Extracts the AI-generated rating from the response (1-5 scale).
        """
        try:
            score = int(response.split("Rating:")[-1].strip().split()[0])
            return min(max(score, 1), 5)
        except Exception:
            return 3  # Default neutral rating if extraction fails

    def store_feedback(self, original, summary, ai_rating, human_rating):
        """
        Stores summary validation history for RLHF.
        """
        feedback_entry = {
            "original": original,
            "summary": summary,
            "ai_rating": ai_rating,
            "human_rating": human_rating
        }
        self.validation_history.append(feedback_entry)
        if self.verbose:
            print(f"[RLHF] Stored AI Rating: {ai_rating}, Human Rating: {human_rating}")

    def optimize_with_rl(self):
        """
        Reinforcement learning: adjust temperature and max_tokens based on feedback trends.
        """
        if len(self.validation_history) < 5:
            return

        ratings = np.array([entry["human_rating"] for entry in self.validation_history])
        avg_rating = np.mean(ratings)

        if avg_rating < 3:
            self.temperature = max(self.temperature - 0.05, 0.3)
        elif avg_rating > 4:
            self.temperature = min(self.temperature + 0.05, 1.0)

        if any(len(entry["summary"]) > self.max_tokens * 0.9 for entry in self.validation_history):
            self.max_tokens = min(self.max_tokens + 50, 1024)

        if self.verbose:
            print(f"[RLHF] Adjusted Ollama settings → Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
