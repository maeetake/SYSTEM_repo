import json
import google.generativeai as genai
import os
import re
from b_QandA import get_json, save_to_json_file

# .envファイルからAPIキーをロード
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set in the environment variables.")

from config import GEMINI_MODEL_NAME as MODEL_NAME

model = genai.GenerativeModel(MODEL_NAME)


def generate_prompt_from_json(user_imput, data):
    """JSONデータからGemini APIのプロンプトを生成する関数"""

    # 基本プロンプトの設定
    prompt = "You are a professional AI model developer.\n"
    prompt += "Based on the user's input requirements provided below, generate a structured specification document outlining the essential components needed to build an AI model. **Adhere strictly to the specified format**.\n\n"

    # ユーザーの入力を追加
    prompt += f"User Input:\n  \"{user_imput}\"\n\n"

    # ユーザー要件を追加
    prompt += "User Requirements:\n"
    for key, value in data.items():
        prompt += f"  - {key}: {value}\n"

    # 固定のフォーマット定義を追加
    prompt += "\n**Output Format:**\n"
    prompt += "The output should follow the format below strictly:\n"
    prompt += "```json"
    prompt += '''{\n'''
    prompt += '''  "model_blueprint": {\n'''
    prompt += '''    "model_role": "A brief description of the AI model's role.",\n'''
    prompt += '''    "instructions": [\n'''
    prompt += '''      "Step-by-step instructions for building the model."\n'''
    prompt += '''    ],\n'''
    prompt += '''    "constraints": [\n'''
    prompt += '''      "A list of constraints or limitations."\n'''
    prompt += '''    ],\n'''
    prompt += '''    "model_details": {\n'''
    prompt += '''      "data_acquisition": [\n'''
    prompt += '''        "Details about obtaining and preparing the required data."\n'''
    prompt += '''        "Note: The data will be provided by the user directly. Do not use APIs, libraries, or external sources to acquire data without explicit instructions from the user."\n'''
    prompt += '''      ],\n'''
    prompt += '''      "data_preprocessing": [\n'''
    prompt += '''        "Details about handling, cleaning, and preprocessing the data."\n'''
    prompt += '''      ],\n'''
    prompt += '''      "model_selection": [\n'''
    prompt += '''        "Details about suitable algorithms and models."\n'''
    prompt += '''      ],\n'''
    prompt += '''      "model_training": [\n'''
    prompt += '''        "Details about the training process."\n'''
    prompt += '''      ],\n'''
    prompt += '''      "model_evaluation": [\n'''
    prompt += '''        "Details about evaluating the model."\n'''
    prompt += '''      ],\n'''
    prompt += '''      "improvement_suggestions": [\n'''
    prompt += '''        "Suggestions for improving the model's accuracy or performance."\n'''
    prompt += '''      ]\n'''
    prompt += '''    }\n'''
    prompt += '''  },\n'''
    prompt += '''  "example": {\n'''
    prompt += '''    "input_data": {\n'''
    prompt += '''      "prompt": "Description of the input data prompt.",\n'''
    prompt += '''      "sample": {}\n'''
    prompt += '''    },\n'''
    prompt += '''    "desired_output": {\n'''
    prompt += '''      "prompt": "Description of the output prompt.",\n'''
    prompt += '''      "sample": {}\n'''
    prompt += '''    },\n'''
    prompt += '''    "accuracy": "Expected accuracy or performance."\n'''
    prompt += '''  }\n'''
    prompt += '''}\n'''
    prompt += "```"



    # 注意事項を追加
    prompt += "\n**Additional Instructions:**\n"
    prompt += "- Adhere to the provided format exactly. Do not deviate from the structure.\n"
    prompt += "- Ensure the output is clear, concise, and includes all necessary details for the user to build the AI model.\n"
    prompt += "- Avoid redundant information.\n\n"

    return prompt


def extract_json_string(text: str) -> str:
    match = re.search(r'{.*}', text, re.DOTALL) # .*に改行を含めるため re.DOTALL を使用
    if match:
        return match.group(0)
    else:
        return ""
    


def call_gemini_api(prompt, model):
    """Gemini APIを呼び出してプロンプトを生成する関数"""

    response = model.generate_content(prompt)
    try:
        out_json = extract_json_string(response.text)
        # Error handling in case the Gemini output is not valid JSON
        summary = json.loads(out_json)
        return json.dumps(summary, indent=4, ensure_ascii=False)
        # return summary
    except json.JSONDecodeError:
        print("Error: Gemini output was not in valid JSON format.")
        print("Gemini Output:\n", response.text)
        return None



def main():
    filename = "1.generated_questions.json"
    user_input = get_json(filename)
    user_input = user_input["user_input"][0]

    filename = "2.structured_data.json"
    json_data = get_json(filename)

    prompt = generate_prompt_from_json(user_input, json_data)

    specification = call_gemini_api(prompt, model)

    if specification:

        programming_recommendations = {
        "preferred_language": "Python.",
        "libraries": "Pandas, NumPy, scikit-learn, TensorFlow or PyTorch."
        }

        specification = json.loads(specification)  
        specification["model_blueprint"]["programming_recommendations"] = programming_recommendations
        specification = specification["model_blueprint"]
        specification = json.dumps(specification, indent=4)

    if specification:
        # print("--- Gemini APIによって生成されたプロンプト ---")
        # print(specification)

        filename = "3.model_specification.json"
        save_to_json_file(specification, filename)

if __name__ == "__main__":
    main()