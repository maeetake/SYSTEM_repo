import os
from openai import OpenAI
from config import OPENAI_MODEL_NAME

# OpenAI APIキーを設定
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 環境変数が設定されていません。")

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_with_openai(prompt: str) -> str:
    """
    Generates content using the OpenAI API.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The OpenAI model to use.

    Returns:
        str: The generated text.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Python code."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
