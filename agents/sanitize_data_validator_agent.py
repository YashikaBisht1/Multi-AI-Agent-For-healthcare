# agents/sanitize_validator_agent.py
from .agent_base import AgentBase
import numpy as np

class SanitizeValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SanitizeValidatorAgent", max_retries=max_retries, verbose=verbose)
        self.validation_history = []
        self.temperature = 0.7
        self.max_tokens = 512

    def execute(self, original_data, sanitized_data):
        """
        Validates PHI removal from sanitized data and applies RLHF on feedback.
        """
        system_msg = "You are an AI that checks if medical data is correctly sanitized (all PHI removed or masked)."
        user_msg = (
            "Evaluate the following:\n\n"
            f"Original:\n{original_data}\n\n"
            f"Sanitized:\n{sanitized_data}\n\n"
            "Make sure the sanitized version replaces PHI using tags like [PATIENT_NAME], [DATE], [LOCATION], etc.\n"
            "Report if any PHI remains and rate the sanitization from 1 to 5 (5 = perfect masking).\n\n"
            "Validation Report:"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        try:
            response = self.call_llama(messages, temperature=self.temperature, max_tokens=self.max_tokens)
            ai_score = self.extract_score(response)
            human_score = self.prompt_human_rating(response)
            self.store_feedback(original_data, sanitized_data, ai_score, human_score)
            self.tune_hyperparams()

            avg_score = round((ai_score + human_score) / 2, 1)
            return response, avg_score

        except Exception as e:
            print(f"[SanitizeValidatorAgent Error] {e}")
            return "Validation failed.", 3.0

    def extract_score(self, response):
        try:
            return min(max(int(response.split("Rating:")[-1].strip().split()[0]), 1), 5)
        except Exception:
            return 3

    def prompt_human_rating(self, response):
        print("\n📋 AI Validation:\n", response)
        while True:
            try:
                score = int(input("🧠 Your Rating (1-5): "))
                if 1 <= score <= 5:
                    return score
            except ValueError:
                pass
            print("⚠️ Please enter a valid number between 1 and 5.")

    def store_feedback(self, original, sanitized, ai, human):
        self.validation_history.append({
            "original": original,
            "sanitized": sanitized,
            "ai_rating": ai,
            "human_rating": human
        })
        if self.verbose:
            print(f"[RLHF] Stored → AI: {ai}, Human: {human}")

    def tune_hyperparams(self):
        if len(self.validation_history) < 5:
            return

        ratings = np.array([entry["human_rating"] for entry in self.validation_history])
        avg = np.mean(ratings)

        if avg < 3:
            self.temperature = max(0.3, self.temperature - 0.05)
        elif avg > 4:
            self.temperature = min(1.0, self.temperature + 0.05)

        if any(len(entry["sanitized"]) > 0.9 * self.max_tokens for entry in self.validation_history):
            self.max_tokens = min(1024, self.max_tokens + 50)

        if self.verbose:
            print(f"[RLHF] New Params → Temp: {self.temperature}, Max Tokens: {self.max_tokens}")
