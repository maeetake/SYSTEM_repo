import os
import json
from typing import List, TypedDict
import google.generativeai as genai
import typing_extensions as typing

# 環境変数からAPIキーを取得
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set in the environment variables.")


# Geminiモデルの設定
from config import MODEL_NAME

# 出力形式の指定
class Keywords_schema(typing.TypedDict):
    keywords: list[str]

class Questions_schema(typing.TypedDict):
    question: list[str]

def create_generation_config(schema):
    return genai.GenerationConfig(
        temperature=1,
        top_p=1,
        top_k=5,
        response_mime_type="application/json",
        response_schema=schema,
    )

GENERATION_CONFIG_KEYWORDS = create_generation_config(Keywords_schema)
GENERATION_CONFIG_QUESTIONS = create_generation_config(Questions_schema)


from config import OUT_JSON_DIR

def save_to_json_file(user_request: str, content: str, filename: str) -> None:
    """
    質問をJSONファイルとして保存する。
    """
    try:
        filename = "1.generated_questions.json"
        OUT_PATH = os.path.join(OUT_JSON_DIR, filename)
        content_dict = json.loads(content)
        content_dict["user_input"] = [user_request]
        with open(OUT_PATH, "w", encoding="utf-8") as file:
            json.dump(content_dict, file, indent=4, ensure_ascii=False)
        print(f"Output saved to {OUT_PATH}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


class InputHandler:
    """
    ユーザーの披掩的なリクエストを受け取り，必要な情報を解析するクラス。
    """

    def __init__(self, model_name=MODEL_NAME):
        self.abstract_request = ""
        self.key_points: List[str] = []
        self.model = genai.GenerativeModel(model_name)

    def receive_request(self, request: str) -> None:
        """
        ユーザーのリクエストを受け取る
        """
        self.abstract_request = request
        print(f"Received request: {self.abstract_request}")

    def analyze_request(self) -> None:
        """
        リクエストを解析して重要なキーワードを抽出する
        """
        prompt = ""
        prompt += "Extract important keywords from the following request."
        prompt += "Keywords should be nouns or compound nouns that are useful for specifying the request."
        prompt += "Request: " + self.abstract_request
        prompt += "Output the keywords as comma-separated values."
        
        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=GENERATION_CONFIG_KEYWORDS,
            )
            self.key_points = json.loads(response.text).get("keywords", [])
            print(f"Extracted keywords: {self.key_points}")
        except Exception as e:
            print(f"Error during keyword extraction: {e}")
            self.key_points = []

class QuestionGenerator:
    """
    キーワードとリクエストに基づいて質問を生成する
    """
    def __init__(self, model_name=MODEL_NAME):
        self.model = genai.GenerativeModel(model_name)

    def generate_questions(self, key_points: List[str], abstract_request: str) -> str:
        """
        キーワードと披掩的なリクエストを基に質問を生成し，JSON文字列として返す。
        """
        if not key_points:
            return json.dumps({"error": "Please provide more details about your request."})

        prompt = ""
        prompt += "List several questions to clarify the user's request and keywords provided below.\n\n"
        prompt += "Request: " + abstract_request + "\n"
        prompt += "Keywords: " + ', '.join(key_points) + "\n\n"
        prompt += "The user intends to develop a system for conducting numerical experiments.\n"
        prompt += "Therefore, design questions that will comprehensively gather all the necessary information to create a numerical experiment system based on the user's request.\n\n"
        prompt += "The questions should be clear and aimed at eliciting specific answers.\n"
        prompt += "The questions should be limited to gathering information necessary to build and train prediction models, as well as visualizing the results. Do not include questions about scalability, system integration, or user interfaces beyond visualization of prediction results.\n"
        prompt += "Output the questions as a JSON list of objects, where the key is 'Question'."

        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=GENERATION_CONFIG_QUESTIONS,
            )
            return self._process_response(response.text, response)
        except Exception as e:
            print(f"Error generating questions: {e}")
            return json.dumps({"error": f"Failed to generate questions: {e}"})

    @staticmethod
    def _process_response(response_text: str, response) -> str:
        """
        モデルのレスポンスを処理してJSON文字列を返す。
        """
        try:
            return json.dumps(json.loads(response_text), indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            print("Error: Could not decode JSON from the response.")
            # print(f"DEBUG\nResponse content: \n{response.text}")
            return json.dumps({"error": "Could not decode JSON from the response."})


def provide_examples() -> None:
    """
    リクエスト例を表示する
    """
    examples = [
        "I want to create a system that predicts NVIDIA's stock price for the next day.",
        "I want to create a program that performs data classification using machine learning.",
        "I want to create a web application with user registration and authentication features.",
    ]
    print("Here are some example requests:")
    for example in examples:
        print(f"  - {example}")



def main() -> None:
    """
    メイン関数: リクエスト受付から質問生成までのワークフローを管理
    """
    print("=== Enter your preference. ===\n")
    print("Examples are ...")

    # 各コンポーネントの初期化
    input_handler = InputHandler()
    question_gen = QuestionGenerator()

    # ユーザーからの入力を取得
    provide_examples()
    user_request = input("\nEnter your request: ")
    input_handler.receive_request(user_request)

    # リクエストの解析
    input_handler.analyze_request()

    # 質問の生成と表示
    questions_json = question_gen.generate_questions(
        input_handler.key_points, input_handler.abstract_request
    )
    print("\n=== Generated Questions (JSON) ===")
    print(questions_json)

    # JSONをファイルに保存
    save_to_json_file(user_request, questions_json, "1.generated_questions.json")

if __name__ == "__main__":
    main()
