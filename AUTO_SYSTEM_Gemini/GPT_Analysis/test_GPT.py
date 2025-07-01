import os
from openai import OpenAI

def generate_text_with_openai(prompt_text: str, model_name: str = "gpt-3.5-turbo") -> str:
    """
    OpenAI API を使用してテキストを生成します。

    Args:
        prompt_text (str): モデルに送信するプロンプト。
        model_name (str): 使用するOpenAIモデルの名前 (例: "gpt-3.5-turbo", "gpt-4o")。

    Returns:
        str: 生成されたテキスト。
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 環境変数が設定されていません。")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=150,  # 生成するトークンの最大数
            temperature=0.7, # 生成のランダム性 (0.0-1.0)
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API の呼び出し中にエラーが発生しました: {e}"

if __name__ == "__main__":
    user_prompt = "Pythonで簡単なWebサーバーを構築する方法を教えてください。"
    generated_response = generate_text_with_openai(user_prompt)
    print("--- 生成されたテキスト ---")
    print(generated_response)

    print("\n--- 別の例 ---")
    another_prompt = "「吾輩は猫である」の冒頭を要約してください。"
    generated_response_2 = generate_text_with_openai(another_prompt, model_name="gpt-4o")
    print(generated_response_2)
