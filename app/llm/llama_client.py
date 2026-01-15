from langchain_ollama import OllamaLLM


class LlamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        max_tokens: int = 300,
    ):
        self.llm = OllamaLLM(
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
        )

    def generate(self, prompt: str) -> str:
        """
        LangChain OllamaLLM은 invoke() 사용
        """
        response = self.llm.invoke(prompt)
        return response
