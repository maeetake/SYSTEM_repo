import os
import anthropic
from config import LLM_MODEL_NAME

# Anthropic APIキーを設定
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY 環境変数が設定されていません。")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_with_llm(prompt: str) -> str:
    """
    Generates content using the Claude API.

    Args:
        prompt (str): The prompt to send to the model.

    Returns:
        str: The generated text.
    """
    message = client.messages.create(
        model=LLM_MODEL_NAME,
        max_tokens=8196, # Claudeの最大トークン数に合わせて調整
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text