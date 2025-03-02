# agents/agent_base.py

import ollama
from abc import ABC, abstractmethod
from loguru import logger
import os

class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_llama(self, messages, temperature=0.7, max_tokens=150):
        """
        Calls the Llama model via Ollama and retrieves the response.

        Args:
            messages (list): A list of message dictionaries.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens in the response.

        Returns:
            str: The content of the model's response.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to Ollama:")
                    for msg in messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")
                
                # Call the Ollama chat API
                response = ollama.chat(
                    model='llama3.2:3b',  # Updated model name
                    messages=messages
                )
                
                # Parse the response to extract the text content
                reply = response['message']['content']
                
                if self.verbose:
                    logger.info(f"[{self.name}] Received response: {reply}")
                
                return reply
            except Exception as e:
                retries += 1
                logger.error(f"[{self.name}] Error during Ollama call: {e}. Retry {retries}/{self.max_retries}")
        raise Exception(f"[{self.name}] Failed to get response from Ollama after {self.max_retries} retries.")
