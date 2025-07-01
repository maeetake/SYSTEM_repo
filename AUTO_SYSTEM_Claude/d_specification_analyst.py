import json
import google.generativeai as genai
import os
import re
import typing_extensions as typing
from config import UNITTEST_DATA_DIR
from b_QandA import get_json, save_to_json_file, extract_json_content
from f_prompt_generator import process_directory


# Google Gemini APIキーを設定
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

from config import GEMINI_MODEL_NAME as MODEL_NAME

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


class task_schema(typing.TypedDict):
    task_name: str
    description: str
    dependencies: list[str]
    constraints: list[str]

class task_list_schima(typing.TypedDict):
    tasks: list[task_schema]

GENERATION_CONFIG = genai.GenerationConfig(
        temperature=1,
        top_p=1,
        top_k=5,
        response_mime_type="application/json",
        response_schema=task_list_schima
)



def analyze_specification(specification_json):
    """
    仕様書を解析し、タスク分解を行う。

    Args:
      specification_json: JSON形式の仕様書文字列

    Returns:
      dict: タスク分解結果を含む辞書
    """
    result = process_directory()

    data_head = result["file_content"]

    specification_json = json.dumps(specification_json, ensure_ascii=False, indent=4)

    prompt = ""
    prompt += "You are an AI assistant that analyzes a given JSON-formatted specification document and decomposes it into tasks required for source code generation.\n"
    prompt += "Read the specification document carefully and extract the following information:\n\n"
    prompt += "1. Model role and purpose.\n"
    prompt += "2. Concrete steps (tasks) that should be executed.\n"
    prompt += "3. Dependencies between tasks.\n"
    prompt += "4. **Essential constraints related to the task's execution, including relevant details from the 'model_details' section. Provide specific details, technologies, and targets. Describe each constraint with 1-2 sentences in a bullet-point list.**\n\n"
    prompt += "A sample of the input data provided is as follows:\n"
    prompt += f"{data_head}"
    prompt += "\n"
    prompt += "Consider the structure and content of this data when implementing."
    prompt += "\n\n"
    prompt += "The output should follow this JSON format:\n\n"
    prompt += "{\n"
    prompt += "    \"tasks\": [\n"
    prompt += "        {\n"
    prompt += "            \"task_name\": \"Name of the task\",\n"
    prompt += "            \"description\": \"**Describe the purpose, specific processes, and expected outcome of the task in 2-4 sentences. Explain why this task is needed in the overall process.**\",\n"
    prompt += "            \"dependencies\": [\"Dependent task name 1\", \"Dependent task name 2\", ...],\n"
    prompt += "            \"constraints\": [\"Constraint 1\", \"Constraint 2\", ...]\n"
    prompt += "        },\n"
    prompt += "        ...\n"
    prompt += "    ]\n"
    prompt += "}\n\n"
    prompt += "Example Output:\n"
    prompt += "{\n"
    prompt += "  \"tasks\": [\n"
    prompt += "    {\n"
    prompt += "      \"task_name\": \"Acquire Historical AAPL Equity Data\",\n"
    prompt += "      \"description\": \"This task involves gathering historical AAPL equity data, which is essential for model training and backtesting. This data will be used to develop a short-term stock price prediction model. Specifically, this task requires to obtain the data from specified data sources and output in JSON format.\",\n"
    prompt += "      \"dependencies\": [],\n"
    prompt += "      \"constraints\": [\"Data sources: Use Alpha Vantage or Tiingo APIs to fetch data.\", \"Data must include: Open, High, Low, Close (OHLC) prices, and volume within the past 10 years.\"]\n"
    prompt += "    },\n"
    prompt += "    {\n"
    prompt += "      \"task_name\": \"Preprocess Data\",\n"
    prompt += "      \"description\": \"This task prepares the raw equity data for model training by addressing missing values, normalizing the data, and engineering new features. The quality of data significantly affects the performance of prediction model, therefore data preprocessing is necessary.\",\n"
    prompt += "      \"dependencies\": [\"Acquire Historical AAPL Equity Data\"],\n"
    prompt += "      \"constraints\": [\"Handle missing values by imputing with the median value.\", \"Normalize the data using MinMaxScaler from sklearn.\", \"Engineer features: Create 5-day and 20-day moving averages, calculate RSI, and MACD using ta-lib.\"]\n"
    prompt += "    }\n"
    prompt += "  ]\n"
    prompt += "}\n\n"
    prompt += "## Specification document:\n"
    prompt += specification_json +"\n"


    # print(prompt)
    # input("="*100)

    response = model.generate_content(
        prompt,
        # generation_config=GENERATION_CONFIG,
        )
    try:
        out_json = extract_json_content(response.text)
        task_decomposition = json.loads(out_json)
    except json.JSONDecodeError:
        print("JSON decode error occurred.")
        print("response:")
        print(response.text)
        return None

    return task_decomposition


def main():

    filename = "3.model_specification.json"
    specification_str = get_json(filename)

    # print(specification_str)

    task_decomposition_result = analyze_specification(specification_str)
    task_decomposition_result = json.dumps(task_decomposition_result, indent=4, ensure_ascii=False)

    if task_decomposition_result:
        filename = "4.task_decomposition.json"
        # print(task_decomposition_result)
        save_to_json_file(task_decomposition_result,filename)
    else:
        print("Failed to decompose tasks.")


if __name__ == '__main__':
    main()