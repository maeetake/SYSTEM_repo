import os
import anthropic

def test_claude_api():
    """
    Claude APIがAPIキーで呼び出せるかテストする簡単なスクリプト。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("エラー: 環境変数 'ANTHROPIC_API_KEY' が設定されていません。")
        print("Claude APIキーを環境変数に設定してください。")
        return

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-3-opus-20240229",  # または利用可能な他のClaudeモデル
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello, Claude!"}
            ]
        )
        print("Claude APIの呼び出しに成功しました。")
        print(f"応答: {message.content}")

    except Exception as e:
        print(f"Claude APIの呼び出し中にエラーが発生しました: {e}")
        print("APIキーが正しいか、またはネットワーク接続を確認してください。")

if __name__ == "__main__":
    test_claude_api()
