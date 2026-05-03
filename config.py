from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llama_cpp_host: str = "127.0.0.1"
    llama_cpp_port: int = 8080
    llama_cpp_api_path: str = "/v1/chat/completions"
    default_model: str = "local"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    stream: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.llama_cpp_host}:{self.llama_cpp_port}"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}{self.llama_cpp_api_path}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
