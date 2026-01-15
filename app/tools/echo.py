from app.tools.base import BaseTool

class EchoTool(BaseTool):
    name = "echo"
    description = "입력 그대로 반환"

    def run(self, text: str) -> str:
        return text
