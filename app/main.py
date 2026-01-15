from app.llm.llama_client import LlamaClient
from app.agents.planner import Planner
from app.agents.executor import Executor
from app.agents.agent import Agent
from app.memory.short_term import ShortTermMemory
from app.tools.registry import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.echo import EchoTool
from app.config.settings import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    MAX_TOKENS,
)


def main():
    # LLM
    llm = LlamaClient(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        max_tokens=MAX_TOKENS,
    )

    planner = Planner(llm)

    # Tool registry
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())

    executor = Executor(registry)
    memory = ShortTermMemory()

    agent = Agent(planner, executor, memory)

    print("🧠 Agent Chatbot 시작 (exit 입력 시 종료)")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            break

        response = agent.run(user_input)
        print(response)


if __name__ == "__main__":
    main()
