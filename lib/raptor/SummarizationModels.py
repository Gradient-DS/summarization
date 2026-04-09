import logging
import os
import sys
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load prompts from pipelines/ if available, otherwise fall back to inline strings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pipelines'))
try:
    from prompts import SUMMARIZATION_SYSTEM, SUMMARIZATION_USER
except ImportError:
    SUMMARIZATION_SYSTEM = "You are a Summarizing Text Portal"
    SUMMARIZATION_USER = "Write a summary of the following, including as many key details as possible: {context}:"

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class CustomPromptSummarizationModel(BaseSummarizationModel):
    """
    Summarization model with caller-supplied system and user prompts.
    Use this to swap in domain-specific prompts (e.g. technical documents)
    without changing the global SUMMARIZATION_SYSTEM / SUMMARIZATION_USER.

    The user_prompt must contain a {context} placeholder.
    """

    def __init__(self, system_prompt: str, user_prompt: str, model: str = "gpt-4.1-nano"):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(context=context)},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class GPT4NanoSummarizationModel(BaseSummarizationModel):
    """Uses gpt-4.1-nano — fast and cost-efficient summarization model."""

    def __init__(self, model="gpt-4.1-nano"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARIZATION_SYSTEM},
                {"role": "user", "content": SUMMARIZATION_USER.format(context=context)},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


