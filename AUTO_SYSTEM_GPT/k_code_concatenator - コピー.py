# import google.generativeai as genai
# import os
# import json
# import re
# import shutil
# import uuid
# import string
# import sys
# from b_QandA import extract_json_content
# from i_code_error_corrector import get_latest_file, get_advice
# from h_code_executor import execute_generated_code
# from g_code_generator import clean_code_fence, save_code, generate_code_2
# from i_code_error_corrector import get_fixed_code, save_noerror_file, error_check
# from j_code_checker import read_file_content, generated_code_evaluation


# # Google Gemini APIキーを設定
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

# from config import MODEL_NAME

# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel(MODEL_NAME)

# def check_concatnated_code(directory, extensions=None):
#     """
#     指定したディレクトリ内にソースコードファイルが存在するかを確認します。

#     Parameters:
#         directory (str): チェックするディレクトリのパス。
#         extensions (list): チェックするファイルの拡張子 (例: ['.py'])。
#                           Noneの場合はデフォルトの拡張子を使用します。

#     Returns:
#         bool: ソースコードファイルが存在する場合はTrue、存在しない場合はFalse。
#     """
#     if extensions is None:
#         extensions = ['.py']

#     # 指定ディレクトリを走査
#     for files in os.walk(directory):
#         for file in files[-1]:
#             if any(file.endswith(ext) for ext in extensions):
#                 return True

#     return False


# import os
# import shutil

# def rename_prefix(number_str):
#     """ 
#     6.1 → a, 6.2 → b のように変換する関数 
#     """
#     mapping = {
#         "6.1": "a", "6.2": "b", "6.3": "c", "6.4": "d",
#         "6.5": "e", "6.6": "f", "6.7": "g", "6.8": "h",
#         "6.9": "i", "6.10": "j"
#     }
#     return mapping.get(number_str, number_str)  # 規定の番号がない場合は変更しない





# def concated_code_evaluation(model_specification_path, python_modules_path, integration_script):
#     """
#     仕様書に対するソースコードの評価を行うためのプロンプトを生成

#     引数:
#         json_specification: 仕様書
#         python_modules : モジュール
#         integration_script: 統合スクリプト

#     戻り値:
#         str: 生成されたプロンプト。
#     """
#     model_specification = read_file_content(model_specification_path)
#     integration_script = read_file_content(integration_script)


#     python_modules = {}
#     # List all files in the directory and filter for Python modules with the specified naming convention
#     module_files = sorted(
#         [f for f in os.listdir(python_modules_path) if f.endswith(".py") and f[0].islower() and f[1:2] == "_"],
#         key=str.lower
#     )
    
#     # Load the content of each filtered Python module
#     for i, module_file in enumerate(module_files, start=1):
#         module_path = os.path.join(python_modules_path, module_file)
#         with open(module_path, 'r', encoding='utf-8') as file:
#             module_content = file.read()
#             module_name = f"Module {i}: {module_file}"
#             python_modules[module_name] = module_content

#     prompt = ""
#     prompt += "#### Evaluation Task\n"
#     prompt += "You are tasked with evaluating the implementation and correctness of the given Python modules and their integration based on a provided JSON specification.\n"
#     # prompt += "Your evaluation should determine if the implemented parts of the code fulfill the tasks described in the JSON. "
#     prompt += "**Only evaluate whether the JSON content is reflected.**"
#     prompt += "If the implementation does not cover the entire JSON specification, **ignore the missing tasks**.\n"
#     prompt += "\n"
#     prompt += "---\n"
#     prompt += "\n"
#     prompt += "### **Files to Evaluate**\n"

#     # Add the JSON specification section
#     prompt += "1. **JSON Specification**  \n"
#     prompt += "```\n"
#     prompt += f"{model_specification}\n"
#     prompt += "```\n"
#     prompt += "\n"

#     # Dynamically add sections for each Python module
#     for i, (module_name, module_code) in enumerate(python_modules.items(), start=2):
#         prompt += f"{i}. **Python Module:** {module_name}  \n"
#         prompt += "**Purpose:** Describe the module's role based on its content or purpose provided.\n"
#         prompt += "```\n"
#         prompt += f"{module_code}\n"
#         prompt += "```\n"
#         prompt += "\n"

#     # Add the integration script section
#     prompt += f"{i+1}. **Integration Script**  \n"
#     prompt += "**Purpose:** Integrate and use the modules  \n"
#     prompt += "```\n"
#     prompt += f"{integration_script}\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "---\n"
#     prompt += "\n"

#     # Guidelines for evaluation
#     prompt += "### **Evaluation Guidelines**\n"
#     prompt += "1. **Coverage Check:** Check if the provided Python code covers the required tasks and constraints from the JSON specification. If only a subset of tasks is implemented, evaluate only the implemented parts and do not comment on or penalize the missing tasks.\n"
#     prompt += "\n"
#     prompt += "2. **Correctness:** Ensure that the provided implementations perform as expected based on the tasks described in the JSON.\n"
#     prompt += "\n"
#     prompt += "3. **Error Handling:** Verify that exceptions and errors are handled correctly, as specified in the JSON constraints.\n"
#     prompt += "\n"
#     prompt += "4. **Input/Output Consistency:** Check if the input/output types and formats of each function match the expectations defined in the JSON. Note: Hardcoded file paths are acceptable and should not be flagged as issues.\n"
#     prompt += "\n"
#     prompt += "5. **Integration:** Ensure that the modules are correctly integrated and that data flow between them works without issues.\n"
#     prompt += "\n"
#     prompt += "---\n"
#     prompt += "\n"

#     # Output format
#     prompt += "### **Output Format (JSON)**\n"
#     prompt += "Provide the evaluation results in the following JSON format:\n"
#     prompt += "\n"
#     prompt += "```\n"
#     prompt += "{\n"
#     prompt += "  \"evaluation\": {\n"
#     prompt += "    \"matches_specification\": \"Yes or No\",\n"
#     prompt += "    \"issues\": [\n"
#     prompt += "      {\n"
#     prompt += "        \"section\": \"Relevant section (e.g., function name, class name, etc.)\",\n"
#     prompt += "        \"description\": \"Description of the issue\",\n"
#     prompt += "        \"suggested_fix\": \"Explanation of recommended fixes\"\n"
#     prompt += "      }\n"
#     prompt += "    ]\n"
#     prompt += "  }\n"
#     prompt += "}\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "## Notes\n"
#     prompt += "- If `matches_specification` is \"Yes\", the `issues` field should be an empty array.\n"

#     # print("="*100)
#     # print(prompt)
#     # input("="*100)

#     response = model.generate_content(
#         prompt,
#         )
    
#     return response.text






# def ensure_directory(path):
#     """ 指定されたディレクトリが存在しない場合は作成する """
#     os.makedirs(path, exist_ok=True)

# def create_init_file(directory):
#     """ 指定されたディレクトリに __init__.py を作成する（存在しない場合）"""
#     init_path = os.path.join(directory, "__init__.py")
#     if not os.path.exists(init_path):
#         with open(init_path, "w", encoding="utf-8"):
#             pass  # 空のファイルを作成

# def rename_file_if_needed(filename):
#     """
#     ファイル名が "X.Y_..." の形式であれば X.Y を変換し、新しいファイル名を返す。
#     それ以外はそのまま返す。
#     """
#     if filename.endswith(".py"):
#         parts = filename.split("_", 1)
#         if len(parts) == 2 and "." in parts[0]:  # "X.Y_..." の形式か確認
#             new_prefix = rename_prefix(parts[0])  # 変換
#             return f"{new_prefix}_{parts[1]}"  # 変更後の名前
#     return filename

# def copy_and_rename(src, dst):
#     """ src の内容を dst にコピーし、必要ならファイル名を変更 """
#     for item in os.listdir(src):
#         src_path = os.path.join(src, item)
#         new_name = rename_file_if_needed(item)
#         dst_path = os.path.join(dst, new_name)

#         if os.path.isdir(src_path):
#             shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
#         else:
#             shutil.copy2(src_path, dst_path)

# def copy_directory():
#     """ ディレクトリ src の全ての要素を dst および dst2 にコピーし、必要に応じてファイル名を変更 """
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     src = os.path.join(current_dir, "NO_ERROR_CODE")
#     dst_list = [
#         os.path.join(current_dir, "CONCATNATED_CODE", "PACKAGE"),
#         os.path.join(current_dir, "CONCATNATED_CODE", "data_maker_init", "PACKAGE"),
#         os.path.join(current_dir, "CONCATNATED_CODE", "HISTORY", "PACKAGE")
#     ]

#     if not os.path.exists(src):
#         raise FileNotFoundError(f"コピー元のディレクトリが見つかりません: {src}")

#     for dst in dst_list:
#         ensure_directory(dst)
#         copy_and_rename(src, dst)
#         create_init_file(dst)


# # ==========

# def find_python_scripts():
#     CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#     directory = os.path.join(CURRENT_DIR, "CONCATNATED_CODE", "PACKAGE")

#     pattern = re.compile(r"^[a-j]_.*\.py$")  

#     # print(directory)
#     matching_files = []
    
#     for _, _, files in os.walk(directory):
#         for file in files:
#             if pattern.match(file):
#                 matching_files.append(file) 

#     return matching_files




# def save_generated_code(code: str, save_dir: str, filename: str = "generated_script.py") -> str:
#     """
#     生成したPythonコードを指定のディレクトリに保存する。

#     Args:
#         code (str): 生成されたPythonコード
#         save_dir (str): 保存先のディレクトリパス
#         filename (str, optional): 保存するファイル名 (デフォルトは 'generated_script.py')

#     Returns:
#         str: 保存されたファイルのフルパス
#     """
#     # ディレクトリが存在しない場合は作成
#     os.makedirs(save_dir, exist_ok=True)

#     # 保存するファイルのフルパス
#     save_path = os.path.join(save_dir, filename)

#     try:
#         # ファイルにコードを書き込む
#         with open(save_path, "w", encoding="utf-8") as file:
#             file.write(code)
#         print(f"Generated code has been saved to: {save_path}")
#         return save_path
#     except Exception as e:
#         print(f"Error saving the file: {e}")
#         return ""





# def generate_first_concatnated_code(script_path: str) -> str:
#     """
#     指定されたPythonスクリプトをモジュールとして利用しながら
#     全く同じ動作をする新しいPythonスクリプトを生成するためのプロンプトを作成する。

#     Args:
#         script_path (str): 元のPythonスクリプトのファイルパス
#                           (例: "C:\\Users\\User\\Documents\\script.py")

#     Returns:
#         str: LLM用のプロンプト
#     """
#     # ファイル名を取得
#     script_filename = os.path.basename(script_path)

#     # モジュール名（拡張子なし）を取得
#     module_name = os.path.splitext(script_filename)[0]

#     # ファイルの内容を取得
#     try:
#         with open(script_path, "r", encoding="utf-8") as file:
#             script_content = file.read()
#     except FileNotFoundError:
#         return f"エラー: 指定されたファイルが見つかりません: {script_path}"
#     except Exception as e:
#         return f"エラー: ファイルを読み込む際に問題が発生しました: {str(e)}"

#     # プロンプトの生成
#     prompt = ""
#     prompt += "You are an expert Python programmer. Create a **single executable Python script** that replicates the functionality of the given Python script without modifications while using it as a module.\n"
#     prompt += "\n"
#     prompt += f"【Original Python Script】\n"
#     prompt += f"Filename: {script_filename}\n"
#     prompt += "```python\n"
#     prompt += f"{script_content}\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "【Requirements】\n"
#     prompt += "1. **Ensure that the new script functions exactly the same as the original script**\n"
#     prompt += "   - The output must be identical.\n"
#     prompt += "   - The same calculations and data processing must be performed.\n"
#     prompt += "\n"
#     prompt += f"2. **If the script defines functions or classes, import them using `from PACKAGE.{module_name} import ...` in the new script**\n"
#     prompt += "   - The script should assume that the module is stored in the `PACKAGE` directory.\n"
#     prompt += "   - Ensure that functions and classes are appropriately called within `if __name__ == \"__main__\":`.\n"
#     prompt += "\n"
#     prompt += f"3. **The output should be a single executable Python script similar to `main.py`**\n"
#     prompt += "   - Use `import` appropriately so that the script can be executed independently.\n"
#     prompt += "   - Ensure that the script can be run directly by using `if __name__ == \"__main__\":`.\n"
#     prompt += "\n"
#     prompt += "【Output Format】\n"
#     prompt += "```python\n"
#     prompt += "# Executable Python script\n"
#     prompt += f"from PACKAGE.{module_name} import {{functions or classes}}\n"
#     prompt += "\n"
#     prompt += "def main():\n"
#     prompt += "    # Call functions from the module\n"
#     prompt += "    {{ Code using the module }}\n"
#     prompt += "\n"
#     prompt += "if __name__ == \"__main__\":\n"
#     prompt += "    main()\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "Generate **a single Python script** that meets these requirements.\n"



#     # print("="*100)
#     # print(prompt)
#     # input("="*100)

#     response = model.generate_content(
#     prompt,
#     )
#     code = clean_code_fence(response.text)

    
#     # response = generate_code(prompt)

#     # code = clean_code_fence(response)

    
#     return code





# def concatnate_code(model_specification_path: str, concatnated_code_path: str, additional_code_path: str) -> str:
#     """
#     model_specification に基づいて、additional_code を concatnated_code に統合する
#     新しい機能を追加せずに、実装済みの部分のみを統合する

#     引数:
#         model_specification_path (str): モデルの仕様（JSONファイル）へのパス
#         concatnated_code_path (str): `concatnated.py` へのパス
#         additional_code_path (str): 追加コードへのパス

#     戻り値:
#         str: LLM用のプロンプト文字列。
#     """


#     model_specification = read_file_content(model_specification_path)
#     concatnated_code = read_file_content(concatnated_code_path)
#     additional_code = read_file_content(additional_code_path)

#     concatnate_code_name = os.path.basename(concatnated_code_path)
#     new_script_name = os.path.basename(additional_code_path)

#     prompt = ""
#     prompt += "Your task is to create a Python program based on the following specifications."
#     prompt += "\n"
#     prompt += "\n"
#     prompt += "## Overview\n"
#     prompt += "Using the following **specifications and existing code**, generate a new integrated source code."
#     prompt += "\n"
#     prompt += "\n"
#     prompt += "### Model Specification\n"
#     prompt += model_specification + "\n"
#     prompt += "\n"
#     prompt += "### Existing Code\n"
#     prompt += f"#### `{concatnate_code_name}` content\n"
#     prompt += "```python\n"
#     prompt += concatnated_code + "\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += f"#### Additional `{new_script_name}` content\n"
#     prompt += "```python\n"
#     prompt += additional_code + "\n"
#     prompt += "```\n"
#     prompt += "\n"

#     prompt += "## Tasks\n"
#     prompt += f"1. Maintain the structure of `{concatnate_code_name}` while integrating the functionality of `{new_script_name}` **as a module**.\n"
#     prompt += f"2. Modify `{concatnate_code_name}` so that it imports necessary functions from `{new_script_name}` using `PACKAGE.` as a prefix.\n"
#     prompt += "3. Follow the specifications, but **limit integration to already implemented parts within the provided scripts, and do not add unimplemented features even if they are mentioned in the specifications**.\n"
#     prompt += "4. Utilize the existing error handling and logging mechanisms to ensure consistency.\n"
#     prompt += "5. Follow **Python best practices for modularity, readability, and maintainability**.\n"
#     prompt += "\n"

#     prompt += "## Constraints\n"
#     prompt += "- **Follow the specifications but do not introduce new functionality beyond what is already implemented.**\n"
#     prompt += "- **Skip parts of the specifications that are not covered in the provided code.**\n"
#     prompt += "- **Only integrate the given scripts without implementing additional requirements.**\n"
#     prompt += f"- **Ensure that `{concatnate_code_name}` imports functions from `{new_script_name}` using `PACKAGE.` as a prefix.**\n"
#     prompt += "- **Ensure that any global variables defined in the __main__ module are not assumed to be available in the imported module's global namespace. Pass necessary data explicitly to functions instead of relying on globals.**\n"
    
    


#     prompt += "- **The generation or use of synthetic data is strictly prohibited. Always use real data based on the specifications.**\n"



    
#     prompt += "\n"


#     packages = find_python_scripts()
#     packages = [s.replace(".py", "") for s in packages]

#     prompt += "## Output Format\n"
#     prompt += "```python\n"
#     for module_name in packages:
#         prompt += f"from PACKAGE.{module_name} import {{functions or classes}}\n"
#     prompt += "{Insert the integrated Python script here}\n"
#     prompt += "```\n"

#     # print("="*100)
#     # print(prompt)
#     # input("="*100)

#     response = model.generate_content(
#     prompt,
#     )
    
#     code = clean_code_fence(response.text)

#     # response = generate_code(prompt)
#     # code = clean_code_fence(response)

#     return code


# def get_latest_alphabet_script(directory: str):
#     # ディレクトリ内の全ファイルを取得
#     files = os.listdir(directory)
    
#     # .pyファイルをフィルタリング
#     py_files = [f for f in files if f.endswith(".py")]
    
#     # アルファベット順で降順ソート
#     py_files.sort(key=lambda x: x[0].lower(), reverse=True)
    
#     return py_files[0] if py_files else None


# def find_max_script_number(directory):
#     pattern = re.compile(r'concatnated_(\d+)\.py$')
#     max_number = -1
#     max_filename = None
    
#     for filename in os.listdir(directory):
#         match = pattern.match(filename)
#         if match:
#             number = int(match.group(1))
#             if number > max_number:
#                 max_number = number
#                 max_filename = filename    
#     if max_number == -1:
#         return None, "concatenated_1.py"
    
#     next_filename = f"concatnated_{max_number + 1}.py"
    
#     return max_filename, next_filename




# def fix_concat_error(model_specification_path, python_modules_path, integration_script_path, evaluation_content):
#     """
#     各入力ファイルと評価フィードバックから，
#     「ダメ出し修正後の複数のPythonモジュールを仕様書に従って組み上げたPythonスクリプト」を出力するプロンプトを生成する関数．
    
#     引数:
#       - model_specification_path: モデル仕様書（JSON）のファイルパス
#       - python_modules_path: 複数のパイソンモジュールが格納されたディレクトリのパス
#       - integration_script_path: 統合済みパイソンスクリプトのファイルパス
#       - evaluation_content: 組み上げたスクリプトに対するダメ出し内容（辞書型）
    
#     戻り値:
#       - プロンプト文字列
#     """

#     concatnate_code_name = os.path.basename(integration_script_path)
#     module_name_all = ""
    
#     # モデル仕様書の読み込みとJSONパース
#     model_specification_str = read_file_content(model_specification_path)
#     try:
#         model_specification = json.loads(model_specification_str)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"モデル仕様書のJSONパースに失敗しました: {e}")
    
#     # 統合スクリプトの読み込み
#     integrated_script = read_file_content(integration_script_path)
    
#     # 複数のパイソンモジュールの読み込み
#     python_modules = {}
#     # ファイル名が「小文字＋アンダースコア～.py」のものを対象とする
#     module_files = sorted(
#         [f for f in os.listdir(python_modules_path) if f.endswith(".py") and f[0].islower() and f[1:2] == "_"],
#         key=str.lower
#     )
    
#     for i, module_file in enumerate(module_files, start=1):
#         module_path = os.path.join(python_modules_path, module_file)
#         with open(module_path, 'r', encoding='utf-8') as file:
#             module_content = file.read()
#             module_name = f"Module {i}: {module_file}"
#             python_modules[module_name] = module_content
#             module_name_all = module_name_all + "  `"+ module_file.removesuffix(".py") +"`"

#     # プロンプトの作成
#     prompt = ""
    
#     # モデル仕様書のセクション
#     prompt += "**Model Specification Requirements**:\n"
#     prompt += f"Role: {model_specification.get('model_role', 'N/A')}\n\n"
#     prompt += "Instructions:\n"
#     if 'instructions' in model_specification:
#         prompt += "\n".join(model_specification['instructions'])
#     prompt += "\n\n"

#     # 統合スクリプトのセクション
#     prompt += "### Current Integrated Script (Assembled According to Specification):\n"
#     prompt += f"**Script Name: {concatnate_code_name}**\n"
#     prompt += "```python\n"
#     prompt += integrated_script + "\n"
#     prompt += "```\n\n"
        
#     # JSON文字列の場合は辞書にパースする
#     if isinstance(evaluation_content, str):
#         try:
#             evaluation_content = json.loads(evaluation_content)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"評価フィードバックのJSONパースに失敗しました: {e}")

#     # 評価フィードバックのセクション
#     prompt += "# Evaluation Feedback Summary:\n"
#     for issue in evaluation_content.get('evaluation', {}).get('issues', []):
#         section = issue.get('section', 'Unknown Section')
#         description = issue.get('description', '')
#         suggested_fix = issue.get('suggested_fix', '')
#         prompt += f"- **Section**: {section}\n"
#         prompt += f"  **Issue**: {description}\n"
#         prompt += f"  **Suggested Fix**: {suggested_fix}\n\n\n"


#     prompt += "## Tasks\n"
#     prompt += f"1. **MANDATORILY REVISE {concatnate_code_name} to INCORPORATE EVERY SINGLE PIECE OF FEEDBACK from the Evaluation Feedback Summary. IT IS ABSOLUTELY ESSENTIAL THAT ALL IDENTIFIED ISSUES ARE COMPLETELY REMEDIATED—NO EXCEPTIONS!**\n"
#     prompt += f"2. Maintain the structure of `{concatnate_code_name}` while integrating the functionality of {module_name_all} **as a module**.\n"
#     prompt += f"3. Modify `{concatnate_code_name}` so that it imports necessary functions from {module_name_all} using `PACKAGE.` as a prefix.\n"
#     prompt += f"4. Modify `{concatnate_code_name}` so that it imports all necessary functions from {module_name_all} consistently using the `PACKAGE.` prefix. In other words, import the module as follows:\n"
#     prompt += "   ```python\n"
#     prompt += f"   from PACKAGE.`module_name` import `functions `\n"
#     prompt += "   ```\n"
#     prompt += "4. Follow **Python best practices for modularity, readability, and maintainability**.\n"


#     # 制約事項
#     prompt += "## Constraints\n"
#     # prompt += "- **Follow the specifications but do not introduce new functionality beyond what is already implemented.**\n"
#     prompt += "- **Skip parts of the specifications that are not covered in the provided code.**\n"
#     prompt += "- **Only integrate the given scripts without implementing additional requirements.**\n"
#     prompt += f"- **Ensure that `{concatnate_code_name}` imports functions from `{module_name_all}` using `PACKAGE.` as a prefix.**\n"



#     # 出力形式の指示
#     prompt += "## Output Format\n"
#     prompt += ("Please provide the **revised integrated Python script** (i.e., the integrated script after applying the fixes "
#                 "based on the evaluation feedback) assembled from the provided Python modules according to the model specification.\n")
#     prompt += "**Return only the corrected code as a complete single block.**\n"
#     prompt += "**Do not provide any explanations. Ensure the code is fully functional.**\n"
#     prompt += "The output should be formatted as follows:\n"
#     prompt += "```python\n"
#     prompt += "# Revised Integrated Script\n"
#     prompt += "{Insert the revised integrated script here}\n"
#     prompt += "```\n\n"

    

#     file_path = r"C:\Users\maeta\OneDrive\デスクトップ\25-02-06\qwen_cleaned\SYSTEM\prompt_for_fixconcat.txt"
#     try:
#         # 書き込みモード('w')でファイルをオープン
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write(prompt)
#         print(f"テキストを正常に {file_path} に保存しました。")
#     except Exception as e:
#         print(f"ファイルの保存中にエラーが発生しました: {e}")

#     # print("="*100)
#     # print(prompt)
#     # input("="*100)




#     response = model.generate_content(
#     prompt,
#     )
    
#     code = clean_code_fence(response.text)

#     # response = generate_code(prompt)

#     # code = clean_code_fence(response)

#     return code


# class ConcatenationError(Exception):
#     """concat後のコードに実行エラーが多すぎた場合の例外"""
#     pass


# def fix_concatnated_code(code, errors, result):

#     prompt = ""
#     prompt += "## Error Fix Request\n"
#     prompt += "\n"
#     prompt += "**Programming Language:** Python\n"
#     prompt += "\n"
#     prompt += "**Code:**\n"
#     prompt += "```python\n"
#     prompt += code + "\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "**Error Message:**\n"
#     prompt += "```\n"
#     prompt += errors + "\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "**Execution Result:**\n"
#     prompt += "```\n"
#     prompt += result + "\n"
#     prompt += "```\n"
#     prompt += "**Please modify the code according to the following requirements:**\n"
#     prompt += "1. **Fix all errors to ensure the code runs without issues.**\n"
#     prompt += "2. **Completely preserve the original functionality. Ensure that no functionality is lost.**\n"
#     prompt += "3. **Return only the corrected code as a complete single block.**\n"
#     prompt += "4. **Do not provide any explanations. Ensure the code is fully functional.**\n\n"
#     prompt += "## Output Format\n"
#     prompt += "**Return only the corrected code as a complete single block.**\n"
#     prompt += "**Do not provide any explanations. Ensure the code is fully functional.**\n"
#     prompt += "The output should be formatted as follows:\n"
#     prompt += "```python\n"
#     prompt += "# Revised Integrated Script\n"
#     prompt += "{Insert the revised integrated script here}\n"
#     prompt += "```\n\n"

   
#     # response  = generate_code(prompt)

#     # code = clean_code_fence(response)

#     response = model.generate_content(
#         prompt,
#         )
    
#     code = clean_code_fence(response.text)
    


#     # print(code)
#     # print("\n^generated code^")
#     # print(type(code))
#     # input("="*100)

#     return code







# def generate_fix_code_simple(source_code: str, execution_output: str) -> str:
#     """
#     Generates a prompt for an LLM to fix the given source code based on execution errors.

#     Parameters:
#         source_code (str): The source code as a string
#         execution_output (str): The execution output, including error messages

#     Returns:
#         str: The prompt to be provided to the LLM
#     """
#     prompt = ""
#     prompt += "## Source Code Error Fix Request\n"
#     prompt += "\n"
#     prompt += "The following source code encountered an error during execution.\n"
#     prompt += "Please fix the error without changing the original functionality of the code.\n"
#     prompt += "**Provide only the fixed code as output without any explanations or comments.**\n"
#     prompt += "\n"
#     prompt += "### Execution Environment\n"
#     prompt += "- Language: Python\n"
#     prompt += "- Expected Behavior: Error-free execution\n"
#     prompt += "\n"
#     prompt += "### Source Code:\n"
#     prompt += "```python\n"
#     prompt += source_code + "\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "### Execution Output (Error Messages):\n"
#     prompt += "```\n"
#     prompt += execution_output + "\n"
#     prompt += "```\n"
#     prompt += "\n"
#     prompt += "### Fix Requirements:\n"
#     prompt += "1. Preserve the original functionality of the source code.\n"
#     prompt += "2. Fix all errors to ensure the code executes correctly.\n"
#     prompt += "3. **The output should be formatted as follows:**\n"
#     prompt += "\n"
#     prompt += "The output should be formatted as follows:\n"
#     prompt += "```python\n"
#     prompt += "# Revised Script\n"
#     prompt += "{Insert the revised script here}\n"
#     prompt += "```\n"
    
#     response = model.generate_content(
#         prompt,
#         )
    
#     code = clean_code_fence(response.text)

#     # print(prompt)
#     # input("pronpt~~~")
    

#     return code




# def main():
#     #　NO＿ERROR＿CODEの内部で，CANCATNATION_CODE/PACKAGE の内部を書き換え
#     # __init__.pyを作成
#     copy_directory()

#     packages = find_python_scripts()
#     print(packages)



#     # PACKAGE の内部 のパッケージ数が1の場合（__init__.py除く）
#     # 今までの連結コードがない場合，（最初）最初のコードだけをモジュールとして引っ張ってきたコードを生成
#     # その１つのパッケージをモジュールとして利用し，機能を再現するソースコードを作成
#     if len(packages) == 1:
#         while True:
#             print("enter_1")
#             CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#             concatnated_code_path = os.path.join(CURRENT_DIR, "CONCATNATED_CODE")
#             package_path = os.path.join(concatnated_code_path, "PACKAGE", packages[0])
            
#             code = generate_first_concatnated_code(package_path) 

#             # print(code)
#             generated_code_path = save_generated_code(code, concatnated_code_path, filename="concatnated_1.py")

#             result, errors = execute_generated_code(generated_code_path)

#             output = result + "\n\n" + errors

#             errors = error_check(output)

#             print(output)
#             print(errors)


#             # エラーがない場合，concatnated.py を上書き
#             if errors.strip() != "Error":

#                 print("concatnated_1.py を 作成")
#                 break

#             else:            
#                 print("===ERROR===")
#                 # print(errors)
#                 # print(errors.strip())
#                 print("再実行します．")
#                 raise ConcatenationError("3回エラーが発生したため処理を中止します")

#     else:
#         CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#         model_specification_path = os.path.join(CURRENT_DIR,"OUT_JSON", "3.model_specification.json")
#         concatnated_code_path = os.path.join(CURRENT_DIR, "CONCATNATED_CODE")
#         max_filename, next_filename = find_max_script_number(concatnated_code_path)
#         concatnated_code_path = os.path.join(concatnated_code_path, max_filename)
#         additional_code_path = os.path.join(CURRENT_DIR, "CONCATNATED_CODE", "PACKAGE")
#         additional_code_path = os.path.join(additional_code_path, get_latest_alphabet_script(additional_code_path))
#         # print("enter else")


#         code = concatnate_code(model_specification_path, concatnated_code_path, additional_code_path)

#         # # print(code)

        
#         # 過去データの保存
#         history_code_path = os.path.join(CURRENT_DIR, "CONCATNATED_CODE", "HISTORY")
#         unique_id = str(uuid.uuid4())
#         history_filename = f"concatnated_{unique_id}.py"
#         generated_code_path = save_generated_code(code, history_code_path, history_filename)



#         print(model_specification_path)
#         print(concatnated_code_path)
#         print(additional_code_path)

#         # input("=*="*100)

#         result, errors = execute_generated_code(generated_code_path)

#         output = result + "\n" + errors

#         llm_check = error_check(output)

#         count = 0        
#         count_2 = 0
#         while True:
#             if llm_check.strip() != "Error":
#                 """
#                 ここでコンキャットスクリプトの評価
#                 """
#                 print(llm_check.strip())

#                 # input("$$"*100)
#                 python_modules_path = os.path.join(CURRENT_DIR, "CONCATNATED_CODE", "PACKAGE")

#                 # print(model_specification_path)
#                 # print(python_modules_path)
#                 # print(generated_code_path)

#                 evaluation = concated_code_evaluation(model_specification_path, python_modules_path, generated_code_path)

#                 evaluation = extract_json_content(evaluation)

#                 print(evaluation)


#                 # file_path = r"C:\Users\maeta\OneDrive\デスクトップ\25-02-06\qwen_cleaned\SYSTEM\___re_concatted_code.txt"
#                 # try:
#                 #     # 書き込みモード('w')でファイルをオープン
#                 #     with open(file_path, 'w', encoding='utf-8') as file:
#                 #         file.write(re_concatted_code)
#                 #     print(f"テキストを正常に {file_path} に保存しました。")
#                 # except Exception as e:
#                 #     print(f"ファイルの保存中にエラーが発生しました: {e}")
                    
#                 # input("&"*100)


#                 # print(type(evaluation))
#                 evaluation_json = json.loads(evaluation)
#                 # print(type(evaluation_json))
#                 result = evaluation_json["evaluation"]["issues"]

#                 """
#                 ここが一生Yesにならない．
#                 """
#                 if result: # 仕様書内容を反映できていなければ
#                     print(result)
#                     count += 1
                    

#                     re_concatted_code = fix_concat_error(model_specification_path, python_modules_path, generated_code_path, evaluation_json)

#                     # print(re_concatted_code)
                


#                     evaluation = concated_code_evaluation(model_specification_path, python_modules_path, generated_code_path)

#                     evaluation = extract_json_content(evaluation)


#                     count_error = 0
#                     while True: # concat 後 & 仕様書内容反映後 のコードのエラー修正ループ

#                         unique_id = str(uuid.uuid4())
#                         history_filename = f"concatnated_{unique_id}.py"
#                         generated_code_path = save_generated_code(re_concatted_code, history_code_path, history_filename)
#                         # input("=*"*100)

#                         result, errors = execute_generated_code(generated_code_path)
#                         # print("####↓を実行####")
#                         # print(generated_code_path)
#                         # print("######################")

#                         print("===RESULT===")
#                         print(result)

#                         print("===ERRORS===")
#                         print(errors)
#                         # input("><"*100)

#                         llm_check = error_check(errors)
#                         print(llm_check)
#                         # input("="*100)
#                         if llm_check.strip() == "Error":

#                             re_concatted_code = fix_concatnated_code(re_concatted_code, errors, result)

#                             llm_check = error_check(errors)
#                             count_error +=1
#                             if count_error == 3:
#                                 print("エラー修正不可能．再実行")
#                                 raise ConcatenationError("3回エラーが発生したため処理を中止します")
#                                 # os.execv(sys.executable, [sys.executable] + sys.argv)
                            
#                         else:
#                             print("仕様書反映後のエラーは無し．仕様書内容を反映")
#                             evaluation = concated_code_evaluation(model_specification_path, python_modules_path, generated_code_path)
#                             print("EVALUATION")
#                             print(evaluation)
#                             # input("="*100)
#                             """
#                             ここのevaluation で Yesならば，break＊２
                            


#                             """
#                             break


#                     if count == 3:
#                         print("仕様書内容を反映できませんでした．")
#                         save_dir_path =  os.path.join(CURRENT_DIR, "CONCATNATED_CODE")
#                         save_generated_code(code, save_dir_path, next_filename)
#                         print("concatnated.py を 更新")
#                         break

#                 else:
#                     save_dir_path =  os.path.join(CURRENT_DIR, "CONCATNATED_CODE")
#                     save_generated_code(code, save_dir_path, next_filename)
#                     print("Yes!!!")
#                     print("concatnated.py を 更新")
#                     break

#             else:            
#                 print("concat 後のコードに実行エラーあり")
#                 if count_2 == 3:
#                     print("無理なのでbreak")
#                     # エラーを明示的に発生させる
#                     raise ConcatenationError("3回エラーが発生したため処理を中止します")
                    
#                 print(llm_check)

#                 print("generated_code")
#                 print(generated_code_path)

#                 source_code = read_file_content(generated_code_path)
#                 execution_output = output

#                 # code = generate_fix_code_simple(source_code, execution_output)

#                 advice = get_advice(source_code, errors, result)

#                 # print(advice)


#                 prompt = ""
#                 prompt += "### Instructions:\n"
#                 prompt += "1. When making corrections, consider the original purpose and intent of the code to ensure it works as intended.\n"
#                 prompt += "2. Pay attention to the input and output formats of the modules being used. If necessary, implement functionality to transform or preprocess the data so that it conforms to the expected formats.\n"
#                 prompt += "3. Ensure that all data processing, model training, and evaluation steps follow logical consistency and provide meaningful results.\n"
#                 specifications = prompt

#                 filepath = get_fixed_code(source_code, errors, result, specifications, advice)

#                 code = read_file_content(filepath)
                
#                 history_code_path = os.path.join(CURRENT_DIR, "CONCATNATED_CODE", "HISTORY")
#                 unique_id = str(uuid.uuid4())
#                 history_filename = f"concatnated_{unique_id}.py"
#                 generated_code_path = save_generated_code(code, history_code_path, history_filename)

#                 result, errors = execute_generated_code(generated_code_path)

#                 print("#"*100)
#                 print(result)
#                 print("===")
#                 print(errors)
#                 print("#"*100)

#                 output = result + "\n" + errors

#                 llm_check = error_check(output)
#                 # input("last")
#                 count_2 += 1



# if __name__ == '__main__':
#     main()