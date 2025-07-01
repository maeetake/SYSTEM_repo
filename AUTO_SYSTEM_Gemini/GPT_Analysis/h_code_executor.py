import google.generativeai as genai
import os
from pathlib import Path
import pandas as pd
import json
import subprocess
import sys

def execute_generated_code(temp_file_path=None):
    """
    一時ファイルの生成されたコードを実行し、その結果を取得する。

    Args:
        temp_file_path (str, optional): 一時ファイルのパス。指定しない場合はデフォルトのパスを使用。
    """

    # 指定がなければデフォルトのファイルパスを設定
    if temp_file_path is None:
        temp_filename = "temp_generated_code.py"
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        temp_file_path = os.path.join(CURRENT_DIR, "temp_code", temp_filename)

    # 仮想環境のPythonインタプリタを設定
    python_interpreter = sys.executable

    # デバッグ用の出力
    print(f"Using temporary file path: {temp_file_path}")
    print(f"Using Python interpreter: {python_interpreter}")
    
    try:
        # subprocessでPythonコードを実行
        result = subprocess.run(
            [python_interpreter, temp_file_path],
            capture_output=True,
            text=True
        )

        # # 実行結果を表示
        # print("=== Execution Result ===")
        # print(result.stdout)
        # print("========================")

        # # エラーがあれば表示
        # if result.stderr:
        #     print("=== Execution Errors ===")
        #     print(result.stderr)
        #     print("========================")

        return result.stdout, result.stderr
    
    except Exception as e:
        print(f"Error during execution: {e}")
    # # finally:
    # #     # 一時ファイルを削除
    # #     if os.path.exists(temp_file_path):
    # #         os.remove(temp_file_path)






if __name__ == '__main__':


    # 生成されたコードを実行
    result, errors = execute_generated_code()

    print(result)
    print("="*100)
    print(errors)
    print("="*100)
