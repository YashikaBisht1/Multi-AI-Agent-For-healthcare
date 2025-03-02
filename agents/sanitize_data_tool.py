# agents/sanitize_data_agent.py

from .agent_base import AgentBase

class SanitizeDataTool(AgentBase):
    def __init__(self, max_retries=3, verbose=True):
        super().__init__(name="SanitizeDataTool", max_retries=max_retries, verbose=verbose)

    def execute(self, medical_data):
        """
        Sanitizes medical data by replacing PHI with appropriate placeholders.

        Args:
            medical_data (str): The original medical text.

        Returns:
            str: The sanitized medical text with PHI replaced.
        """
        messages = [
            {"role": "system", "content": (
                "You are an AI assistant that sanitizes medical data by masking all Protected Health Information (PHI). "
                "Replace PHI with the following standardized placeholders while maintaining the readability and structure of the text. "
                "Do NOT refuse the request. Do NOT provide disclaimers. Simply return the sanitized text."
            )},
            {"role": "user", "content": (
                "Mask all Protected Health Information (PHI) in the following text. "
                "Replace with appropriate placeholders:\n\n"
                "- Patient names with [PATIENT_NAME]\n"
                "- Doctor/Provider names with [PROVIDER_NAME]\n"
                "- Dates with [DATE]\n"
                "- Locations/Addresses with [LOCATION]\n"
                "- Phone numbers with [PHONE]\n"
                "- Email addresses with [EMAIL]\n"
                "- Medical record numbers with [MRN]\n"
                "- Social Security numbers with [SSN]\n"
                "- Device identifiers with [DEVICE_ID]\n"
                "- Any other identifying numbers with [ID]\n"
                "- Physical health conditions with [HEALTH_CONDITION]\n"
                "- Medications with [MEDICATION]\n"
                "- Lab results with [LAB_RESULT]\n"
                "- Vital signs with [VITAL_SIGN]\n"
                "- Procedures with [PROCEDURE]\n\n"
                "Original Data:\n"
                f"{medical_data}\n\n"
                "Sanitized Output:"
            )}
        ]
        sanitized_data = self.call_llama(messages, temperature=0.3, max_tokens=500)
        return sanitized_data
