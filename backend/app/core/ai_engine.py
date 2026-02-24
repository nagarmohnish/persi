import logging
from anthropic import AsyncAnthropic
from app.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None


def get_ai_client() -> AsyncAnthropic | None:
    global _client
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def chat_completion(
    system_prompt: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request to Claude.

    Args:
        system_prompt: The system prompt with stage-aware context.
        messages: List of {"role": "user"|"assistant", "content": str} dicts.
        model: Claude model to use.
        max_tokens: Max response tokens.

    Returns:
        The assistant response text, or a fallback message if API unavailable.
    """
    client = get_ai_client()
    if not client:
        return "AI responses are not available — please set ANTHROPIC_API_KEY in your .env file."

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"AI chat completion failed: {e}")
        return f"I encountered an error generating a response. Please try again. (Error: {type(e).__name__})"


async def generate_summary(text: str, summary_type: str = "meeting") -> str:
    """Generate a summary of the given text."""
    client = get_ai_client()
    if not client:
        return "AI summary unavailable — set ANTHROPIC_API_KEY."

    prompts = {
        "meeting": "Summarize this meeting transcript. Include: key decisions, action items, follow-ups, and main topics discussed.",
        "note": "Create a concise summary of this note, highlighting the key insights and action items.",
        "general": "Provide a clear, concise summary of the following text.",
    }

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"{prompts.get(summary_type, prompts['general'])}\n\n{text}"}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"AI summary generation failed: {e}")
        return f"Summary generation failed. (Error: {type(e).__name__})"
