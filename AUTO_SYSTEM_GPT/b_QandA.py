import os
import json
from typing import Optional, Any
import google.generativeai as genai
import re

# Gemini APIの設定 (APIキーは環境変数に設定してください)
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set in the environment variables.")

from config import GEMINI_MODEL_NAME as MODEL_NAME

# Gemini Proモデルのロード
model = genai.GenerativeModel(MODEL_NAME)

GENERATION_CONFIG = genai.GenerationConfig(
        temperature=1,
        top_p=1,
        top_k=5,
    )

from config import OUT_JSON_DIR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_json(filename: str) -> Optional[Any]:
    """
    指定されたJSONファイルを読み込み、Pythonデータ構造として返却する。

    Args:
        filename (str): 読み込むJSONファイルの名前。

    Returns:
        Optional[Any]: JSONデータ (辞書またはリストなど) が読み込めた場合はそのデータ、
                      それ以外の場合は None を返す。
    """
    file_path = os.path.join(OUT_JSON_DIR, filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。パス: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"エラー: '{filename}' は有効なJSONファイルではありません。")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None


def save_to_json_file(data, filename):
    """
    構造化データをスクリプトのディレクトリにあるJSONファイルに保存

    param data： param data: 保存する構造化データ
    param filename: JSONファイルの名前
    """
    try:
        # Get the directory of the current script
        file_path = os.path.join(OUT_JSON_DIR, filename)

        # Write data to the file
        with open(file_path, "w", encoding="utf-8") as json_file:
            # json.dump(data, json_file, indent=4, ensure_ascii=False)
            json_file.write(data)

        print(f"Structured data saved to {file_path}")
    except Exception as e:
        print(f"Error saving structured data to file: {e}")


def extract_json_content(text):
    """
    入力文字列中の最初の「```json」から最後の「```」までを取得し，返す．

    Parameters:
        text (str): 入力文字列

    Returns:
        str: 最初の「```json」から最後の「```」までの間にある内容．
             該当部分が見つからない場合は空文字列
    """
    try:
        # 正規表現で最初の「```json」から最後の「```」を抽出
        pattern = r'```json(.*)```'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return text


def collect_responses(questions):
    """
    ユーザーからの一連の質問に対する回答を収集

    :param questions： param questions: ユーザーへの質問のリスト
    :return： 構造化された回答を含む辞書
    """
    structured_data = {}

    print("Please answer the following questions:")

    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")
        response = input("Your answer: ").strip()

        structured_data[f"Q{index}"] = {
            "question": question,
            "answer": response
        }
    return structured_data


def summarize_structured_data(structured_data):
    """
    構造化データからキーと値のペアを抽出し、JSON形式でまとめる

    引数
        structured_data (dict)： 質問と回答を含む構造化データ

    戻り値
        str： JSON形式の要約
    """

    prompt = f"""
        Given the structured data, extract key-value pairs that represent the answers to the questions and output them in JSON format.
        Use keys that best represent the content of the answers.

        Example:
        Input:
        {{
            'Q1': {{'question': 'What is the specific time frame you are interested in for stock price predictions (e.g., intraday, daily, weekly, monthly)?', 'answer': 'I would like to forecast on a daily basis.'}},
            'Q2': {{'question': 'Which specific stocks or stock indices are you interested in predicting?', 'answer': 'I would like to make a prediction mainly for NASDAQ.'}}
        }}
        Output:
        {{
            "time_frame": "daily",
            "stocks_or_indices": "NASDAQ"
        }}

        Input:
        {json.dumps(structured_data)}
        Output:
        """
    prompt = ""
    prompt += "Given the structured data, extract key-value pairs that represent the answers to the questions and output them in JSON format."
    prompt += "Use keys that best represent the content of the answers."
    prompt += "Example:\n"
    prompt += "Input:\n"
    prompt += "{\n"
    prompt += '    "Q1": {\n'
    prompt += '        "question": "What is the specific time frame you are interested in for stock price predictions (e.g., intraday, daily, weekly, monthly)?",\n'
    prompt += '        "answer": "I would like to forecast on a daily basis.",\n'
    prompt += '    },\n'
    prompt += '    "Q2": {\n'
    prompt += '        "question": "Which specific stocks or stock indices are you interested in predicting?",\n'
    prompt += '          "answer": "I would like to make a prediction mainly for NASDAQ."\n'
    prompt += '    }\n'
    prompt += '}\n\n'
    prompt += "Output:\n"
    prompt += "{\n"
    prompt += '    "time_frame": "daily",\n'
    prompt += '    "stocks_or_indices": "NASDAQ"\n'
    prompt += "}\n\n"
    prompt += "Input:\n"
    prompt += json.dumps(structured_data,indent=4)
    prompt += "\nOutput:"
        

    response = model.generate_content(
        contents=prompt,
        generation_config=GENERATION_CONFIG,
    )
    try:
        out_json = extract_json_content(response.text)
        summary = json.loads(out_json)
        return json.dumps(summary, indent=4, ensure_ascii=False)
    except json.JSONDecodeError:
        print("Error: Gemini output was not in valid JSON format.")
        print("Gemini Output:\n", response.text)
        return None




def main():
    filename = "1.generated_questions.json"
    questions = get_json(filename)

    # Collect responses from the user
    responses = collect_responses(questions["question"])

    # Categorize responses
    categorized_data = summarize_structured_data(responses)

    # Save the categorized responses to a JSON file
    filename = "2.structured_data.json"
    save_to_json_file(categorized_data, filename)

if __name__ == "__main__":
    main()
