import json
import os
from b_QandA import get_json
from pathlib import Path
import pandas as pd

from config import UNITTEST_DATA_DIR, CODE_PROMPT_DIR

def read_file(file_path: str) -> str:
    """
    ファイル内容を上から6行だけ取得する。
    対応フォーマット: txt, csv, json
    """
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return "\n".join([line.strip() for _, line in zip(range(6), f)])
        elif ext == ".csv":
            df = pd.read_csv(file_path, nrows=6)
            return df.to_string(index=False)
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return json.dumps(data[:6], indent=2, ensure_ascii=False)
                elif isinstance(data, dict):
                    return json.dumps(dict(list(data.items())[:6]), indent=2, ensure_ascii=False)
                else:
                    raise ValueError("JSON形式が不正です")
        else:
            return "対応していないファイル形式です"
    except Exception as e:
        return f"エラー: {str(e)}"

def process_directory(use_generated=False):
    """
    指定されたディレクトリ内で最も新しく作成された対応形式のファイルを処理し、
    結果を｛ファイルパス, ファイルの上から6行｝の形式で返す。
    対応フォーマットでない場合はファイルパスのみ返す。
    """
    if use_generated:
        directory_path = os.path.join(UNITTEST_DATA_DIR, "generated")
    else:
        directory_path = UNITTEST_DATA_DIR
    directory = Path(directory_path)
    if not directory.is_dir():
        return "指定されたパスはディレクトリではありません。"

    # ディレクトリ内のすべてのファイルを取得
    all_files = list(directory.glob("*"))
    if not all_files:
        return "ファイルが見つかりませんでした。"

    # 最も新しく作成されたファイルを特定
    newest_file = max(all_files, key=lambda f: f.stat().st_mtime)

    # 拡張子をチェックして対応フォーマットかを確認
    supported_extensions = [".txt", ".csv", ".json"]
    if newest_file.suffix.lower() in supported_extensions:
        # 対応フォーマットの場合は内容を処理
        file_content = read_file(str(newest_file))
    else:
        # 対応フォーマットでない場合は内容は空にする
        file_content = ""

    # 結果をフォーマットして返す
    result = {
        "file_path": str(newest_file),
        "file_content": file_content
    }

    return result


# Function to generate prompts for each module
def generate_prompts_for_modules(modules, model_spec, use_generated=False):
    prompts = {}

    result = process_directory(use_generated)

    data_path = result["file_path"]
    data_head = result["file_content"]

    for module_name, module_details in modules.items():
        # Extract details from the module
        model_role_and_purpose = module_details.get("model_role_and_purpose", "")
        concrete_tasks = module_details.get("concrete_tasks", "")
        input_specs = module_details.get("constraints", {}).get("input_formats_and_data_types", "")
        output_specs = module_details.get("constraints", {}).get("output_formats_and_data_types", "")
        library_versions = module_details.get("constraints", {}).get("library_versions_and_configurations", "")
        error_handling = module_details.get("constraints", {}).get("specific_error_handling", "")
        code_skeleton = module_details.get("code_skeleton", "")

        prompt = ""
        prompt += f"## Generate Python Code for Module: {module_name}\n\n"
        prompt += "Please generate Python module code based on the detailed specifications below. Ensure strict adherence to all specified requirements and accurately reflect the provided examples. If there are any ambiguities or contradictions in the specifications, use comments in the code to explain the reasoning and adopt a solution that prioritizes functionality and clarity.\n\n"

        prompt += "**1. Model Overview:**\n\n"
        prompt += f"* **Role and Purpose:** {model_role_and_purpose}\n"
        prompt += "* **Overall Specification:**\n"
        prompt += "```json\n"
        prompt += f"{model_spec}\n"
        prompt += "```\n\n"

        prompt += "**2. Module Definition:**\n\n"
        prompt += f"* **Name:** {module_name}\n"
        prompt += "* **Concrete Tasks:**\n"
        prompt += f"    {concrete_tasks}\n\n"

        prompt += "**3. Input/Output Specifications:**\n\n"
        prompt += "* **Input Format (with examples):**\n"
        prompt += f"    {input_specs}\n"
        prompt += "* **Data Head (example):**\n"
        prompt += "```\n"
        prompt += f"{data_head}\n"
        prompt += "```\n"
        prompt += "* **Data Path:**\n"
        prompt += f"    {data_path}\n"
        prompt += "* **Expected Output Format (with examples):**\n"
        prompt += f"    r'{output_specs}'\n\n"

        prompt += "**4. Constraints and Dependencies:**\n"
        prompt += "* **Library Versions and Configurations:**\n"
        prompt += f"    {library_versions}\n"
        prompt += "* **Error Handling Requirements (specific errors to handle and how):**\n"
        prompt += f"    {error_handling}\n\n"

        prompt += "**5. Code Skeleton (if applicable):**\n"
        prompt += f"    {code_skeleton}\n\n"

        prompt += "**6. Implementation Guidelines:**\n\n"

        prompt += "* **Data Retrieval Instructions:** Implement the process to load the dataset using the provided `data_path`. "
        prompt += "Adapt the retrieval method to the data source (e.g., CSV, JSON, database, or API). "
        prompt += "Ensure the program checks the validity of `data_path` and automatically loads the data without user intervention. "
        prompt += "If the file is missing or corrupted, handle errors with informative messages. Optionally, provide mock data for testing purposes to allow the program to proceed. "
        prompt += "```python\n"
        prompt += "import pandas as pd\n"
        prompt += "import os\n\n"
        prompt += "def load_data(data_path: str):\n"
        prompt += "    if not os.path.exists(data_path):\n"
        prompt += "        raise FileNotFoundError(f\"Error: File not found at {data_path}\")\n"
        prompt += "    try:\n"
        prompt += "        # Adjust loading logic based on file type\n"
        prompt += "        if data_path.endswith('.csv'):\n"
        prompt += "            return pd.read_csv(data_path)\n"
        prompt += "        elif data_path.endswith('.json'):\n"
        prompt += "            return pd.read_json(data_path)\n"
        prompt += "        else:\n"
        prompt += "            raise ValueError(\"Unsupported file format. Please provide a CSV or JSON file.\")\n"
        prompt += "    except Exception as e:\n"
        prompt += "        raise ValueError(f\"Error loading data: {e}\")\n"
        prompt += "```\n"

        prompt += "* **main Function Instructions:** Implement a main function as the program's entry point. "
        prompt += "The main function should be designed to execute without requiring any user input during runtime. "
        prompt += "Use the provided `data_path` to automatically load the necessary data. If the data is missing or the path is incorrect, handle errors with informative messages and provide mock data as a fallback if specified. "
        prompt += "```python\n"
        prompt += "def main():\n"
        prompt += "    # Use default data path to automatically load data\n"
        prompt += f"    data_path = '{data_path}'\n"
        prompt += "\n"
        prompt += "    # Load the data using the specified data path\n"
        prompt += "    try:\n"
        prompt += "        df = load_data(data_path)\n"
        prompt += "        print(\"Data loaded successfully. Head of the data:\")\n"
        prompt += "        print(df.head())\n"
        prompt += "    except FileNotFoundError:\n"
        prompt += "        print(\"Data file not found. Please check the data path or provide valid data.\")\n"
        prompt += "        # Optional: Use mock data if necessary\n"
        prompt += "        mock_data = pd.DataFrame({\"column1\": [0, 1, 2], \"column2\": [\"sample1\", \"sample2\", \"sample3\"]})\n"
        prompt += "        print(\"Using mock data:\")\n"
        prompt += "        print(mock_data.head())\n"
        prompt += "    except Exception as e:\n"
        prompt += "        print(f\"An error occurred: {e}\")\n"
        prompt += "```\n"

        prompt += "* **Clarity and Readability:** Prioritize clear, concise, and well-commented code. Use descriptive variable names and function names.\n"
        prompt += "* **Modularity and Reusability:** Design the code for modularity and reusability, considering potential future extensions.\n"
        prompt += "* **Thorough Error Handling:** Implement robust error handling as specified in the `Error Handling Requirements` section, using appropriate exception classes and informative error messages. Include try-except blocks where necessary.\n"
        prompt += "* **Dependency Management:** If the module requires specific external libraries, clearly indicate these dependencies as comments at the beginning of the code. For example: `# Requires: pandas >= 1.0.0`\n"
        prompt += "* **Adherence to Specifications:** Ensure the generated code strictly adheres to the provided specifications. If any adjustments are necessary due to conflicting or ambiguous information, clearly document the rationale behind these changes in comments.\n"
        prompt += "* **Type Hinting (Recommended):** Use type hinting to improve code readability and maintainability.\n\n"

        prompts[module_name] = prompt



    return prompts

def save_text_to_file(i, text_data, module_name):
    """
    テキストデータを指定した相対パスとファイル名で保存する関数。

    Args:
        i (int): ファイルインデックス
        text_data (str): 保存するテキストデータ。
        module_name (str): モジュール名。
    """
    file_name = f"6.{i}_{module_name}_prompt.txt"
    os.makedirs(CODE_PROMPT_DIR, exist_ok=True)
    file_path = os.path.join(CODE_PROMPT_DIR, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text_data)
    print(f"テキストデータを {file_path} に保存しました。")

def main():
    filename_model = "3.model_specification.json"
    filename_modules = "5.defined_modules.json"
    model_specification = get_json(filename_model)
    defined_modules = get_json(filename_modules)

    model_specification_json = json.dumps(model_specification, indent=4)

    # ユーザーからインデックス番号を入力
    print("実行するモジュールのインデックス番号を指定してください（1から）。全て実行する場合は 0 を入力してください。")
    selected_index = int(input("インデックス番号: "))

    print("生成済みデータを使用しますか？ (yes/no)")
    use_generated_input = input("入力: ").lower()
    use_generated = use_generated_input == 'yes'

    # Generate prompts for each module
    prompts = generate_prompts_for_modules(defined_modules, model_specification_json, use_generated)

    # Save prompts based on selected index
    for i, (module_name, prompt) in enumerate(prompts.items(), start=1):
        if selected_index == 0 or selected_index == i:
            save_text_to_file(i, prompt, module_name)

if __name__ == "__main__":
    main()