# agents/write_article_validator_agent.py
import numpy as np
from .agent_base import AgentBase
class WriteArticleValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="WriteArticleValidatorAgent", max_retries=max_retries, verbose=verbose)
        self.validation_history = []
        self.temperature = 0.7
        self.max_tokens = 512

    def execute(self, topic, article):
        system_message = "You are an AI assistant that validates research articles."
        user_content = (
            "Given the topic and the article, assess whether the article comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
            "Provide a brief analysis and rate the article on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
            f"Topic: {topic}\n\n"
            f"Article:\n{article}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation_response = self.call_llama(messages, temperature=self.temperature, max_tokens=self.max_tokens)

        ai_rating = self.extract_validation_score(validation_response)
        human_rating = self.get_human_feedback(validation_response)
        self.store_feedback(topic, article, ai_rating, human_rating)
        self.optimize_with_rl()

        return validation_response

    def extract_validation_score(self, response):
        try:
            score = int(response.split("Rating:")[-1].strip().split()[0])
            return min(max(score, 1), 5)
        except Exception:
            return 3

    def get_human_feedback(self, response):
        print("\n🔍 AI Validation Response:")
        print(response)
        while True:
            try:
                rating = int(input("🤖 Please rate this validation (1-5): "))
                if 1 <= rating <= 5:
                    return rating
                else:
                    print("❌ Invalid input. Enter a number between 1 and 5.")
            except ValueError:
                print("❌ Invalid input. Enter a numeric value.")

    def store_feedback(self, topic, article, ai_rating, human_rating):
        feedback_entry = {
            "topic": topic,
            "article": article,
            "ai_rating": ai_rating,
            "human_rating": human_rating
        }
        self.validation_history.append(feedback_entry)
        if self.verbose:
            print(f"[RLHF] Stored AI Rating: {ai_rating}, Human Rating: {human_rating}")

    def optimize_with_rl(self):
        if len(self.validation_history) < 5:
            return

        ratings = np.array([entry["human_rating"] for entry in self.validation_history])
        avg_rating = np.mean(ratings)

        if avg_rating < 3:
            self.temperature = max(self.temperature - 0.05, 0.3)
        elif avg_rating > 4:
            self.temperature = min(self.temperature + 0.05, 1.0)

        if any(len(entry["article"]) > self.max_tokens * 0.9 for entry in self.validation_history):
            self.max_tokens = min(self.max_tokens + 50, 1024)

        if self.verbose:
            print(f"[RLHF] Adjusted Settings → Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
