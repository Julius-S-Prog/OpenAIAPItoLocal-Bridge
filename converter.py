from models import Message, LlamaCppRequest, ChatCompletionRequest


def format_messages(messages: list[Message]) -> list[dict]:
    """
    Convert OpenAI-format messages to llama.cpp format.
    Only keeps role and content, ignores extra fields.
    """
    formatted = []
    for msg in messages:
        formatted.append({
            "role": msg.role,
            "content": msg.content,
        })
    return formatted


def build_llama_request(openai_request: ChatCompletionRequest) -> LlamaCppRequest:
    """
    Build a llama.cpp request from an OpenAI request.
    Ignores model name, keeps only relevant generation parameters.
    """
    return LlamaCppRequest(
        messages=format_messages(openai_request.messages),
        temperature=openai_request.temperature
        if openai_request.temperature is not None
        else 0.7,
        n_predict=openai_request.max_tokens
        if openai_request.max_tokens is not None
        else 2048,
        top_p=openai_request.top_p if openai_request.top_p is not None else 0.95,
        stream=openai_request.stream,
        stop=openai_request.stop,
        frequency_penalty=openai_request.frequency_penalty,
        presence_penalty=openai_request.presence_penalty,
    )
