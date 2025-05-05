from .agent_base import AgentBase
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatbotAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True, use_biogpt=True):
        super().__init__("ChatbotAgent", max_retries=max_retries, verbose=verbose)
        self.use_biogpt = use_biogpt

        if self.use_biogpt:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")

    def execute(self, user_input):
        if self.use_biogpt:
            inputs = self.tokenizer(user_input, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        else:
            # Fall back to llama or your previous method
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for medical research."},
                {"role": "user", "content": user_input}
            ]
            return self.call_llama(messages)
