

# agents/agent_base.py

import ollama
from abc import ABC, abstractmethod
from loguru import logger

class AgentBase(ABC):
    def __init__(self, name, model='llama3.2:3b', max_retries=2, verbose=True):
        """
        Base class for all agents.

        Args:
            name (str): Agent name (used for logging).
            model (str): Name of the Ollama model to use.
            max_retries (int): Number of retry attempts for API calls.
            verbose (bool): Whether to enable verbose logging.
        """
        self.name = name
        self.model = model
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_llama(self, messages, temperature=0.7,max_tokens=512):
        """
        Calls the Llama model via Ollama and retrieves the response.

        Args:
            messages (list): A list of message dictionaries with 'role' and 'content'.
            temperature (float): Sampling temperature.

        Returns:
            str: The model's response content.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to Ollama ({self.model}):")
                    for msg in messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")

                # Call Ollama's chat API
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": temperature}
                )

                # Extract and return the response content
                reply = response.get("message", {}).get("content", "").strip()

                if not reply:
                    raise ValueError("Received empty response from Ollama.")

                if self.verbose:
                    logger.info(f"[{self.name}] Response: {reply}")

                return reply

            except Exception as e:
                retries += 1
                logger.error(f"[{self.name}] Ollama error: {e} (Retry {retries}/{self.max_retries})")

        raise RuntimeError(f"[{self.name}] Failed to get response from Ollama after {self.max_retries} retries.")
