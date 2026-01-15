class Executor:
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def execute(self, plan: dict) -> str:
        if plan.action == "response":
            return plan.content or ""

        if plan.action == "tool":
            tool = self.tool_registry.get(plan.tool_name)
            return tool.run(**(plan.args or {}))

        return "Invalid action"
