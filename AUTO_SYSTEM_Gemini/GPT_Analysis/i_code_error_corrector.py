from openai_client import generate_with_openai
import os
from pathlib import Path
import re
import shutil
from g_code_generator import clean_code_fence, save_code, generate_code_2
from h_code_executor import execute_generated_code
from config import TEMP_CODE_DIR, NO_ERROR_CODE_DIR, CODE_PROMPT_DIR



def get_fixed_code(code, errors, result, specifications, advice):

    prompt = ""
    prompt += "## Error Fix Request\n"
    prompt += "\n"
    prompt += "**Programming Language:** Python\n"
    prompt += "\n"
    prompt += "**Code:**\n"
    prompt += "```python\n"
    prompt += code + "\n"
    prompt += "```\n"
    prompt += "\n"
    prompt += "**Error and Result:**\n"
    prompt += "```\n"
    prompt += errors + "\n\n"
    prompt += result + "\n"
    prompt += "```\n"
    prompt += "\n"

    if specifications is not None:
        # prompt += "**Specifications:**\n"
        prompt += specifications + "\n\n"

    prompt += "**Please modify the code according to the following requirements:**\n"
    prompt += "1. **Fix all errors to ensure the code runs without issues.**\n"
    prompt += "2. **Completely preserve the original functionality. Ensure that no functionality is lost.**\n"
    prompt += "3. **Return only the corrected code as a complete single block.**\n"
    prompt += "4. **Do not provide any explanations. Ensure the code is fully functional.**\n"
    prompt += "\n"

    prompt += "**Modification Guidelines:**\n"
    prompt += "* **Clarity and Readability:** Prioritize clear, concise, and well-commented code. Use descriptive variable names and function names.\n"
    prompt += "* **Modularity and Reusability:** Design the code for modularity and reusability, considering potential future extensions.\n"
    prompt += "* **Thorough Error Handling:** Implement robust error handling as specified in the `Error Handling Requirements` section, using appropriate exception classes and informative error messages. Include try-except blocks where necessary.\n"
    prompt += "* **Dependency Management:** If the module requires specific external libraries, clearly indicate these dependencies as comments at the beginning of the code. For example: `# Requires: requests==2.25.1`\n"
    prompt += "* **Type Hinting (Recommended):** Use type hinting to improve code readability and maintainability.\n"
    prompt += "* **Ensure Shape Compatibility:** Before performing operations on arrays or data frames, check their shapes and ensure compatibility. If any mismatch is detected, raise a warning. For example, include checks like `assert data.shape[0] == target.shape[0]` to prevent dimension mismatches.\n"
    
    if specifications is None:
        prompt += "* **Main Execution Context:** The generated code should include a `main()` function for execution. Ensure that the following pattern is followed:\n"
        prompt += "```python\n"
        prompt += "def main():\n"
        prompt += "    # Read data from CSV or other sources\n"
        prompt += "    data = pd.read_csv('path/to/your/data.csv')\n"
        prompt += "    result = {function_name}(data)\n"
        prompt += "    print(result.head())\n"
        prompt += "```\n"
        prompt += "  This ensures that the script runs its main logic only when executed directly and does not execute when imported as a module.\n"

    prompt += "\n"
    prompt += "## Detailed Error Analysis and Fix Instructions\n"
    prompt += advice

    response = generate_with_openai(prompt)
    
    code = clean_code_fence(response)

    filepath = save_code(code)

    return filepath

def get_latest_file(directory_path=None):
    if directory_path is None:
        directory_path = TEMP_CODE_DIR
    
    directory = Path(directory_path)

    if not directory.is_dir():
        return "指定されたパスはディレクトリではありません。"
    
    files = [file for file in directory.iterdir() if file.is_file()]
    if not files:
        return "ディレクトリ内にファイルがありません。"
    
    latest_file = max(files, key=lambda file: file.stat().st_mtime)
    return str(latest_file)

def save_noerror_file(file_path):
    directory_path = CODE_PROMPT_DIR
    directory = Path(directory_path)
    files = [file for file in directory.iterdir() if file.is_file()]
    latest_file = max(files, key=lambda file: file.stat().st_mtime)
    file_name = os.path.basename(latest_file)
    file_name = file_name.replace("-", "_")
    file_name = file_name.replace(",", "_")
    file_name = os.path.splitext(file_name)[0] + ".py"
    print(file_name)
    target_directory = NO_ERROR_CODE_DIR
    os.makedirs(target_directory, exist_ok=True)

    new_file_path = os.path.join(target_directory, file_name)
    
    try:
        shutil.copy(file_path, new_file_path)
        print(f"ファイルを以下に保存しました: {new_file_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def error_check(output: str) -> str:
    """
    ソースコード実行後の出力を解析し、それがエラーかどうかを判定するプロンプトを生成する関数。

    Parameters:
        output (str): ソースコード実行後の出力

    Returns:
        str: 生成されたプロンプト
    """
    prompt = ""
    prompt += "You are an assistant that analyzes the output of a program and determines whether it contains an error.\n"
    prompt += "Follow these rules when analyzing the output:\n"
    prompt += "1. **Definition of Error:**\n"
    prompt += "    - If the output contains **critical failure indicators** such as `Error`, `Exception`, `Traceback`, `Segmentation fault`, or phrases like `an error occurred`, it is considered an error.\n"
    prompt += "    - Ignore messages labeled as `INFO` or `WARNING`. These are not considered errors unless the message explicitly mentions an unrecoverable failure.\n"
    prompt += "    - Ignore common system messages related to optimizations, floating-point precision, or CPU instructions.\n"
    prompt += "    - Warnings or minor anomalies (e.g., related to environment configurations) should not be considered errors.\n"
    prompt += "\n"
    prompt += "2. **Classification of Output:**\n"
    prompt += "   - `Error`: The program did not complete successfully and requires a fix. Only classify as `Error` if a critical failure is clearly indicated.\n"
    prompt += "   - `Warning`: The program can run, but the output contains messages requiring attention (such as optimization warnings or system configuration messages).\n"
    prompt += "   - `Normal`: The program completed successfully without any errors or warnings.\n"
    prompt += "\n"
    prompt += "Determine whether the following output should be classified as `Error`, `Warning`, or `Normal`, and return **only one of these three words**.\n"
    prompt += "Do not provide any additional explanations or reasoning.\n"
    prompt += "\n"
    prompt += "Output:\n"
    prompt += "```plaintext\n"
    prompt += output.strip() + "\n"
    prompt += "```\n"
    prompt += "\n"
    prompt += "Return only one of the following: `Error`, `Warning`, or `Normal`.\n"

    response = generate_with_openai(prompt)

    return response

class ConcatenationError(Exception):
    """concat後のコードに実行エラーが多すぎた場合の例外"""
    pass

def get_advice(source_code: str, error_message: str, execution_output: str) -> str:
    """
    This function generates a prompt template that instructs an LLM to provide concrete and practical
    advice for error fixing by combining the basic principles of prompt engineering.
    
    The prompt includes clear instructions, context, input data, and output format requirements.
    """
    prompt = ""
    prompt += "You are an experienced programmer.\n"
    prompt += "Based on the information provided below, please analyze the error and offer concrete, practical advice to fix the source code.\n"
    prompt += "\n"
    prompt += "[Source Code]\n"
    prompt += source_code + "\n"
    prompt += "\n"
    prompt += "[Error Message]\n"
    prompt += error_message + "\n"
    prompt += "\n"
    prompt += "[Execution Output]\n"
    prompt += execution_output + "\n"
    prompt += "\n"
    prompt += "Using the above information, please address the following points:\n"
    prompt += "1. Analyze the cause of the error.\n"
    prompt += "2. Identify the specific parts of the code that need to be modified and explain how to fix the error.\n"
    prompt += "3. Provide additional test cases or further explanations if necessary.\n"
    
    prompt += "4. Ensure that input and output data formats are compatible between functions or modules. If necessary, suggest appropriate data conversion methods to meet the expected formats."
    prompt += "5. Verify that the proposed changes maintain consistency across the codebase and do not introduce new issues in related parts of the program."
    
    prompt += "\n"
    prompt += "For each proposed modification, list the affected line numbers along with the updated code examples, and explain your reasoning step-by-step.\n"
    
    response = generate_with_openai(prompt)

    return response

def main():
    count = 0
    filepath = get_latest_file()
    result, errors = execute_generated_code(filepath)

    output = result + "\n" + errors
    print(output)
    llmout = error_check(output)

    while True:        
        if llmout.strip() == "Error":

            temp_file_path = get_latest_file()

            try:
                if os.path.exists(temp_file_path):
                    with open(temp_file_path, 'r', encoding='utf-8') as file:
                        code = file.read()
                else:
                    print(f"ファイル '{temp_file_path}' は存在しません。")
            except Exception as e:
                print(f"ファイルを読み取る際にエラーが発生しました: {e}")

            specification_path = CODE_PROMPT_DIR
            specifications = get_latest_file(specification_path)
            
            try:
                if os.path.exists(specifications):
                    with open(specifications, 'r', encoding='utf-8') as file:
                        specifications = file.read()
                else:
                    print(f"ファイル '{specifications}' は存在しません。")
            except Exception as e:
                print(f"ファイルを読み取る際にエラーが発生しました: {e}")

            advice = get_advice(code, errors, result)

            filepath = get_fixed_code(code, errors, result, None, advice)

            result, errors = execute_generated_code(filepath)

            print(result)
            print("\n\n===ERROR===\n")
            print(errors)
            print("="*100)
            output = result + "\n\n" + errors
            llmout = error_check(output)
            print(llmout)
            
            if llmout.strip() != "Error":
                print("All errors fixed")
                save_noerror_file(filepath)
                break
            
            else:
                count += 1
                print("count = " + str(count) + "/3")
                if count == 3:
                    print("無理なのでbreak")
                    raise ConcatenationError("3回エラーが発生したため処理を中止します")
                
        else:
            print("No errors")
            save_noerror_file(filepath)
            break

if __name__ == '__main__':
    main()