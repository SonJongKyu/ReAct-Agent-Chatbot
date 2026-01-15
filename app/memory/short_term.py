class ShortTermMemory:
    def __init__(self, max_turns: int = 5):
        self.history = []
        self.max_turns = max_turns

    def add(self, user, agent):
        self.history.append(f"User: {user}\nAgent: {agent}")
        self.history = self.history[-self.max_turns:]

    def get(self):
        return "\n".join(self.history)
