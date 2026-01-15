class BaseTool:
    name: str = ""
    description: str = ""

    def run(self, **kwargs) -> str:
        raise NotImplementedError
