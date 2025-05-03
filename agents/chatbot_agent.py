from .agent_base import AgentBase
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatbotAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True, use_biogpt=True):
        super().__init__("ChatbotAgent", max_retries=max_retries, verbose=verbose)
        self.use_biogpt = use_biogpt

        if self.use_biogpt:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large")

    def execute(self, user_input):
        if self.use_biogpt:
            prompt = f"<human>: {user_input}\n<bot>:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove echo and only return the bot response
            if "<bot>:" in decoded:
                response = decoded.split("<bot>:")[-1].strip()
            else:
                response = decoded.strip()

            return response

        else:
            # Fall back to llama or another assistant
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant for patients and researchers."},
                {"role": "user", "content": user_input}
            ]
            return self.call_llama(messages)
