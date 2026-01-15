from app.tools.base import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "수식 계산"

    def run(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception as e:
            return f"계산 오류: {e}"
