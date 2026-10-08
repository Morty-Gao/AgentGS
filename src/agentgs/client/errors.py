class ProviderError(RuntimeError):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"LLM HTTP {status_code}")
