import json
import numpy as np
from .agent_base import AgentBase

class SanitizeDataValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SanitizeDataValidatorAgent", max_retries=max_retries, verbose=verbose)
        self.validation_history = []  # Store validation feedback
        self.temperature = 0.7  # Initial temperature
        self.max_tokens = 512  # Initial max token limit

    def execute(self, original_data, sanitized_data):
        """
        Validates PHI removal using structured masking rules and collects RLHF feedback.
        """
        system_message = (
            "You are an AI assistant that validates the sanitization of medical data by checking for PHI removal. "
            "Your task is to ensure that all Protected Health Information (PHI) has been masked correctly."
        )

        user_content = (
            "Review the original and sanitized data. Identify any remaining PHI and assess the effectiveness of sanitization. "
            "Ensure that all PHI follows the masking guidelines:\n\n"
            "- Patient names → [PATIENT_NAME]\n"
            "- Doctor/Provider names → [PROVIDER_NAME]\n"
            "- Dates → [DATE]\n"
            "- Locations/Addresses → [LOCATION]\n"
            "- Phone numbers → [PHONE]\n"
            "- Email addresses → [EMAIL]\n"
            "- Medical record numbers → [MRN]\n"
            "- Social Security numbers → [SSN]\n"
            "- Device identifiers → [DEVICE_ID]\n"
            "- Any other identifying numbers → [ID]\n"
            "- Physical health conditions → [HEALTH_CONDITION]\n"
            "- Medications → [MEDICATION]\n"
            "- Lab results → [LAB_RESULT]\n"
            "- Vital signs → [VITAL_SIGN]\n"
            "- Procedures → [PROCEDURE]\n\n"
            "Check if any PHI remains unmasked in the sanitized data.\n"
            "Provide a detailed report listing any PHI detected and rate the sanitization process on a scale of 1 to 5, "
            "where **5 indicates complete and accurate sanitization.**\n\n"
            f"Original Data:\n{original_data}\n\n"
            f"Sanitized Data:\n{sanitized_data}\n\n"
            "Validation Report:"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        # Call Ollama with dynamic temperature & token settings
        validation_response = self.call_llama(
            messages, temperature=self.temperature, max_tokens=self.max_tokens
        )

        # Extract AI rating from response
        ai_rating = self.extract_validation_score(validation_response)

        # Get human feedback
        human_rating = self.get_human_feedback(validation_response)

        # Store feedback for reinforcement learning
        self.store_feedback(original_data, sanitized_data, ai_rating, human_rating)

        # Adjust Ollama's parameters based on feedback
        self.optimize_with_rl()

        return validation_response

    def extract_validation_score(self, response):
        """
        Extracts the AI-generated validation score (1-5) from the model's response.
        """
        try:
            score = int(response.split("Rating:")[-1].strip().split()[0])
            return min(max(score, 1), 5)  # Ensure score is between 1 and 5
        except Exception:
            return 3  # Default neutral rating if extraction fails

    def get_human_feedback(self, response):
        """
        Requests human feedback for validation and returns a manual rating.
        """
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

    def store_feedback(self, original, sanitized, ai_rating, human_rating):
        """
        Stores validation history with human feedback for RLHF.
        """
        feedback_entry = {
            "original": original,
            "sanitized": sanitized,
            "ai_rating": ai_rating,
            "human_rating": human_rating
        }
        self.validation_history.append(feedback_entry)

        if self.verbose:
            print(f"[RLHF] Stored AI Rating: {ai_rating}, Human Rating: {human_rating}")

    def optimize_with_rl(self):
        """
        Uses RLHF to adjust temperature & max tokens dynamically.
        """
        if len(self.validation_history) < 5:  # Require enough feedback before tuning
            return

        # Convert stored data into training-like examples
        ratings = np.array([entry["human_rating"] for entry in self.validation_history])

        # Adjust temperature based on rating trends
        avg_rating = np.mean(ratings)
        if avg_rating < 3:  # If validation is poor, reduce randomness for precision
            self.temperature = max(self.temperature - 0.05, 0.3)
        elif avg_rating > 4:  # If validation is excellent, allow more flexibility
            self.temperature = min(self.temperature + 0.05, 1.0)

        # Adjust max_tokens if responses are cut off
        if any(len(entry["sanitized"]) > self.max_tokens * 0.9 for entry in self.validation_history):
            self.max_tokens = min(self.max_tokens + 50, 1024)

        if self.verbose:
            print(f"[RLHF] Adjusted Ollama settings → Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
