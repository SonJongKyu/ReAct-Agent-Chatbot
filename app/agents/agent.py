# from app.agents.base import BaseAgent

# class Agent(BaseAgent):
#     def __init__(self, planner, executor, memory):
#         self.planner = planner
#         self.executor = executor
#         self.memory = memory

#     def run(self, user_input: str) -> str:
#         context = self.memory.get()
#         plan = self.planner.plan(user_input, context)
#         result = self.executor.execute(plan)
#         self.memory.add(user_input, result)
#         return result
