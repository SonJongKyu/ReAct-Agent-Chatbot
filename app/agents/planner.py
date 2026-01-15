from app.llm.prompt_templates import PLANNER_PROMPT
from app.utils.json_parser import parse_json
from app.schemas.agent_action import AgentAction


class Planner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, user_input: str, context: str) -> dict:
        prompt = PLANNER_PROMPT.format(
            user_input=user_input,
            context=context
        )
        raw = self.llm.generate(prompt)
        parsed = parse_json(raw)
        return AgentAction(**parsed)
