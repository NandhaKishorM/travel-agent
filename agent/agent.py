from pathlib import Path

from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from agent.tracer import covert_to_trace

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

client = AsyncOpenAI(max_retries=3)
model = OpenAIModel("gpt-4o", provider=OpenAIProvider(openai_client=client))
agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)



class TravelAgent:
    def __init__(self, model_name=None):
        self.agent = agent
        if model_name:
            self.agent.model._model_name = model_name

    def run(self, query: str):
        response = self.agent.run_sync(query)
        return response.output, covert_to_trace(response.all_messages())