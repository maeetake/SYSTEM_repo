"""
concat後のコードを実行
実行結果とコードを基にプロンプト生成
生成したプロンプトを基に，データ保存用コードを生成
生成したコードを実行し，データを保存

加えて上記のエラー修正
"""

import google.generativeai as genai
import os
import json
import re
import shutil
import subprocess
from b_QandA import extract_json_content
from i_code_error_corrector import get_latest_file, error_check
from h_code_executor import execute_generated_code
from g_code_generator import clean_code_fence, save_code
from i_code_error_corrector import get_fixed_code, save_noerror_file
from k_code_concatenator import find_max_script_number
from j_code_checker import read_file_content
import sys

# Google Gemini APIキーを設定
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

from config import MODEL_NAME

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)




def generate_outdata(script_path: str,execution_result: str,  save_path: str) -> str:
    """
    Generates a prompt for an LLM to modify the given source code by adding a feature
    to save execution results to a specified directory.

    Args:
        source_code (str): The original source code.
        save_path (str): The directory path where the file should be saved (e.g., "./output/").

    Returns:
        str: The prompt for the LLM.
    """

    # ファイルの内容を取得
    try:
        with open(script_path, "r", encoding="utf-8") as file:
            source_code = file.read()
    except FileNotFoundError:
        return f"エラー: 指定されたファイルが見つかりません: {script_path}"
    except Exception as e:
        return f"エラー: ファイルを読み込む際に問題が発生しました: {str(e)}"
    
    directory_path = CODE_PROMPT_DIR
    code_prompt_path = get_latest_file(directory_path)
    code_prompt = read_file_content(code_prompt_path)


    prompt = ""
    prompt += "Please modify the following source code to add a feature that **saves the input data expected by the module following source_code as CSV** based on the provided specification.\n"
    prompt += "- From the content of the specification document, determine the input data that the next module (e.g., the module named `train_time-series_models`) expects, and save that data as a properly formatted CSV file.\n"
    prompt += f"- **Hardcode the file save directory as `{save_path}`.**\n"
    prompt += "- **The file name and extension must be appropriately determined by the LLM (for example, `expected_input.csv`).**\n"
    prompt += "- **Ensure that the CSV file includes all required details as specified in the specification document, including headers and all necessary rows and columns.**\n"
    prompt += "- **Save the file only if the expected input data can be correctly determined based on the specification. If not, do not save any file.**\n"
    prompt += "- **Do not change the function's purpose (do not modify its arguments, return value, or core logic, and avoid unnecessary changes).**\n"
    prompt += "- **Add only the saving functionality without affecting the existing features.**\n"
    prompt += "- Maintain the original structure of the code as much as possible.\n"
    prompt += "- Ensure readability of the code while incorporating the necessary file processing steps.\n"
    prompt += "- Implement appropriate error handling.\n"
    prompt += "\n"
    prompt += "## Source Code Usage Documentation:\n"
    prompt += "Below is the specification document that describes the expected input data for the module following source_code:\n"
    prompt += "```text\n"
    prompt += code_prompt
    prompt += "```\n"
    prompt += "\n"
    prompt += "## Input:\n"
    prompt += "```python\n"
    prompt += source_code + "\n"
    prompt += "```\n"
    prompt += "\n"
    prompt += "## Additional Requirements:\n"
    prompt += f"- **Add a process to create the directory if necessary to ensure that the directory `{save_path}` exists.**\n"
    prompt += "- **Determine the file name and extension appropriately, and hardcode them in the code.**\n"
    prompt += "- **If the expected input data is determined based on the specification, use `to_csv()` to save it as CSV, ensuring that all required headers, index (if applicable), and all rows and columns are included.**\n"
    prompt += "- **If the expected input data is not properly determined, do not save the file.**\n"
    prompt += "- **Implement proper error handling for cases such as the directory not existing or errors during the write operation.**\n"
    prompt += "- **Ensure that only one file is saved per execution, and that previous results are overwritten.**\n"
    prompt += "- **Do not modify the function's arguments, return value, or existing logic.**\n"
    prompt += "- **If necessary, you may add a new function outside of the existing function for handling the saving process.**\n"
    prompt += "\n"
    prompt += "## Output:\n"
    prompt += "Provide only the modified source code.\n"
    prompt += "Generate a **single Python script** that meets these requirements.\n"
    prompt += "Add comments as needed to indicate the changes.\n"


    # print("="*100)
    # print(prompt)
    # input("="*100)

    response = model.generate_content(
    prompt,
    )
    
    code = clean_code_fence(response.text)

    return code




def play_code(file_path):
    """
    生成されたコードを実行し、その結果を取得する。

    Args:
        file_path (str): 実行ファイルのパス。指定しない場合はデフォルトのパスを使用。
    """

    # 指定がなければデフォルトのファイルパスを設定
    if file_path is None:
        temp_filename = "temp_generated_code.py"
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(CURRENT_DIR, "temp_code", temp_filename)

    # 仮想環境のPythonインタプリタを設定
    python_interpreter = sys.executable

    # # デバッグ用の出力
    # print(f"Using temporary file path: {file_path}")
    # print(f"Using Python interpreter: {python_interpreter}")
    
    try:
        # subprocessでPythonコードを実行
        result = subprocess.run(
            [python_interpreter, file_path],
            capture_output=True,
            text=True
        )

        # 実行結果を表示
        print("=== Execution Result ===")
        print(result.stdout)
        print("========================")


        llm_check = error_check(result.stdout)

        if llm_check.strip() == "Error":

            print("=== Execution Errors ===")
            print(result.stderr)
            print("ここでエラーが出た場合の修正処理は未実装（concat でのエラー）")
            print("警告の場合もここに表示される．")
            # input("="*100)

        return result.stdout, result.stderr
    
    except Exception as e:
        print(f"Error during execution: {e}")



import uuid
def save_code_for_data(code):
    """
    生成されたコードを一時ファイルに保存

    Args:
        code (str): 実行するPythonコード
    """
    unique_id = str(uuid.uuid4())
    temp_filename = f"temp_generated_code_{unique_id}.py"

    code_path = os.path.join(DATA_MAKER_INIT_DIR, temp_filename)
    os.makedirs(DATA_MAKER_INIT_DIR, exist_ok=True)

    # 一時ファイルにコードを書き込む
    with open(code_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(code)

    return code_path

class ConcatenationError(Exception):
    """concat後のコードに実行エラーが多すぎた場合の例外"""
    pass


from config import CONCATNATED_CODE_DIR, UNITTEST_DATA_DIR, CODE_PROMPT_DIR, DATA_MAKER_INIT_DIR

def main():
    count = 0
    search_dir = CONCATNATED_CODE_DIR
    max_filename, next_filename = find_max_script_number(search_dir)
    concatnated_code_path = os.path.join(CONCATNATED_CODE_DIR, max_filename)

    save_path = os.path.join(UNITTEST_DATA_DIR, "generated")

    # concat 後のコードを実行
    result, error = play_code(concatnated_code_path)

    # 実行結果とともに，コード入力，出力生成するコード作成
    code = generate_outdata(concatnated_code_path, result, save_path)

    # コードを一時ファイルとして保存
    code_path = save_code_for_data(code)

    # 保存したコードを実行し，データセットを生成
    result, error = play_code(code_path)
    
    output = result + "\n" + error

    llm_check = error_check(output)

    if llm_check.strip() == "Error":

        print("===ERROR===")
        print(error.strip())
        print("データ生成用コードでのエラー発生時の処理は未実装")
        print("警告の場合もここに表示")
        if count == 3:
            print("無理なのでbreak")
            # エラーを明示的に発生させる
            raise ConcatenationError("3回エラーが発生したため処理を中止します")
        count += 1

    else: # エラーがない場合，concatnated.py を上書き

        print("no_error")
    """
    データ生成用コードでの
    エラー発生時の処理は未実装
    """
    

if __name__ == '__main__':
    main()