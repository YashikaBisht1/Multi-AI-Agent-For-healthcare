# agents/write_article_agent.py

import numpy as np
from .agent_base import AgentBase

class WriteArticleTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="WriteArticleTool", max_retries=max_retries, verbose=verbose)
        self.article_history = []  # Store feedback history
        self.temperature = 0.7  # Initial temperature
        self.max_tokens = 1000  # Initial max token limit

    def execute(self, topic, outline=None):
        system_message = "You are an expert academic writer."
        user_content = f"Write a research article on the following topic:\nTopic: {topic}\n\n"
        if outline:
            user_content += f"Outline:\n{outline}\n\n"
        user_content += "Article:\n"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        article = self.call_llama(messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return article

    def store_feedback(self, topic, article, ai_rating, human_rating):
        feedback_entry = {
            "topic": topic,
            "article": article,
            "ai_rating": ai_rating,
            "human_rating": human_rating
        }
        self.article_history.append(feedback_entry)
        if self.verbose:
            print(f"[RLHF] Stored AI Rating: {ai_rating}, Human Rating: {human_rating}")

    def optimize_with_rl(self):
        if len(self.article_history) < 5:
            return  # Need enough feedback before tuning

        ratings = np.array([entry["human_rating"] for entry in self.article_history])
        avg_rating = np.mean(ratings)

        if avg_rating < 3:
            self.temperature = max(self.temperature - 0.05, 0.3)
        elif avg_rating > 4:
            self.temperature = min(self.temperature + 0.05, 1.0)

        if any(len(entry["article"]) > self.max_tokens * 0.9 for entry in self.article_history):
            self.max_tokens = min(self.max_tokens + 100, 2048)

        if self.verbose:
            print(f"[RLHF] Adjusted Settings → Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
