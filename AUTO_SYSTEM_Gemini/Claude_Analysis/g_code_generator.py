import re
import os
import uuid
from config import CODE_PROMPT_DIR, TEMP_CODE_DIR
from llm_client import generate_with_llm

def generate_code_2(prompt):
    return generate_with_llm(prompt)

def read_specific_text_file(file_index):
    """
    指定したインデックスに基づいて特定のテキストファイルを取得し、その内容を表示する。

    Args:
        file_index (int): ファイルインデックス（例: 1, 2 など）
    """
    directory_path = CODE_PROMPT_DIR

    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    target_prefix = f"6.{file_index}"

    for filename in os.listdir(directory_path):
        if filename.startswith(target_prefix) and filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    print(f"File: {filename}")
                    return content
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}")
                return

    print(f"No file found with prefix '{target_prefix}' in directory '{directory_path}'.")

def clean_code_fence(code):
    """
    コードフェンス（```python や ```）を削除する。
    """
    try:
        pattern = r'```python\s*(?:\r?\n)(.*?)(?:\r?\n)?```'
        match = re.search(pattern, code, re.DOTALL)
        return match.group(1).strip() if match else code
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return code

def save_code(code):
    """
    生成されたコードを一時ファイルに保存
    """
    unique_id = str(uuid.uuid4())
    temp_filename = f"temp_generated_code_{unique_id}.py"

    temp_file_path = os.path.join(TEMP_CODE_DIR, temp_filename)
    os.makedirs(TEMP_CODE_DIR, exist_ok=True)

    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(code)

    return temp_file_path

def main():
    print("実行するモジュールのインデックス番号を指定してください（1から）。全て実行する場合は 0 を入力してください。")
    file_index = int(input("インデックス番号: "))
    prompt = read_specific_text_file(file_index)

    if prompt:
        out_code = generate_code_2(prompt)
        cleaned_code = clean_code_fence(out_code)
        save_code(cleaned_code)

if __name__ == '__main__':
    main()
