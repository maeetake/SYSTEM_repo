# from curses import OK
import google.generativeai as genai
import os
import json
import re
from b_QandA import extract_json_content
from i_code_error_corrector import get_latest_file, error_check
from h_code_executor import execute_generated_code
from g_code_generator import clean_code_fence, save_code, generate_code_2
from i_code_error_corrector import get_fixed_code, save_noerror_file, get_advice

# Google Gemini APIキーを設定
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

from config import MODEL_NAME

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_file_content(file_path):
    """
    指定されたファイルパスからファイルを読み取り、内容を出力します。

    Args:
        file_path (str): 読み取るファイルのパス。

    Returns:
        str: ファイルの内容（成功した場合）。
        None: ファイルが存在しない場合やエラーが発生した場合。
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # print(f"ファイル '{file_path}' の内容:\n")
                # print(content)
                return content
        else:
            print(f"ファイル '{file_path}' は存在しません。")
            return None
    except Exception as e:
        print(f"ファイルを読み取る際にエラーが発生しました: {e}")
        return None
    


def generated_code_evaluation(specification: str, source_code: str) -> str:
    """
    Generates a prompt for evaluating source code against a specification document.

    Args:
        specification (str): The text of the specification document.
        source_code (str): The text of the source code to evaluate.

    Returns:
        str: The generated prompt.
    """
    prompt = ""
    prompt += "### Code Evaluation Instructions\n"
    prompt += "\n"
    prompt += "#### Instructions\n"
    prompt += "Compare the following specification document and source code to evaluate whether the source code has been correctly implemented according to the requirements specified in the document. Provide the evaluation results in JSON format.\n"
    prompt += "- **Skip the evaluation of any part of the specification that is not covered by the code provided.**\n"
    prompt += "- **Evaluate only implemented sections.**\n"
    prompt += "- **Do not evaluate content that is planned to be implemented elsewhere.**\n"
    prompt += "- Do not make evaluations regarding `Overall Implementation`."
    prompt += "\n"
    prompt += "**You do not need to consider anything that is not explicitly stated in the specification document. Only evaluate based on the specified requirements.**\n"
    prompt += "\n"
    prompt += "**Do not evaluate the library versions used in the source code. Library versioning should be ignored during the evaluation.**\n"
    prompt += "\n"




    prompt += "- **Even if the specification states that synthetic data should not be used, it is allowed in this case as long as the synthetic data is generated within the main function.**\n"





    prompt += "#### Specification Document\n"
    prompt += "Evaluate based on the content of the specification document provided below. The specification includes requirements such as the purpose of the functions, inputs, outputs, processing details, and error handling.\n"
    prompt += "\"\"\"\n"
    prompt += f"{specification}\n"
    prompt += "\"\"\"\n"
    prompt += "\n"
    prompt += "#### Source Code\n"
    prompt += "Compare the following source code against the specification document to evaluate whether it adheres to the specified requirements.\n"
    prompt += "\n"
    prompt += "```python\n"
    prompt += f"{source_code}\n"
    prompt += "```\n"
    prompt += "\n"

    prompt += "#### Evaluation Criteria\n"
    prompt += "Evaluate based on the following criteria and provide the results in JSON format:\n"
    prompt += "\n"
    prompt += "1. **Compliance with the Specification**\n"
    prompt += "   - Determine whether the source code is implemented as specified (Yes / No).\n"
    prompt += "\n"
    prompt += "2. **Corrections** (only if the answer is No)\n"
    prompt += "   - Provide a description of the issues.\n"

    prompt += "\n"
    prompt += "#### Output Format (JSON)\n"
    prompt += "Provide the evaluation results in the following JSON format:\n"
    prompt += "\n"
    prompt += "\"\"\"\n"
    prompt += "{\n"
    prompt += "  \"evaluation\": {\n"
    prompt += "    \"matches_specification\": \"Yes or No\",\n"
    prompt += "    \"issues\": [\n"
    prompt += "      {\n"
    prompt += "        \"section\": \"Relevant section (e.g., function name, class name, etc.)\",\n"
    prompt += "        \"description\": \"Description of the issue\"\n"
    prompt += "        \"suggested_fix\": \"Explanation of recommended fixes\"\n"
    prompt += "      }\n"
    prompt += "    ],\n"
    prompt += "  }\n"
    prompt += "}\n"
    prompt += "\"\"\"\n"
    prompt += "\n"
    prompt += "#### Notes\n"
    prompt += "- If `matches_specification` is \"Yes\", the `issues` field should be an empty array.\n"
    
    # print("="*100)
    # print(prompt)
    # input("="*100)

    response = model.generate_content(
        prompt,
        )
    
    return response.text


def code_generation(source_code: str, json_str: str, code_prompt) -> str:
    """
    Generates a prompt for modifying a source code based on issues extracted from a JSON string.

    Args:
        source_code (str): The original source code to be modified.
        json_str (str): A JSON string containing evaluation results with issues.

    Returns:
        str: The generated prompt.
    """

    # Parse the JSON string
    evaluation_data = json.loads(json_str)
    issues = evaluation_data.get("evaluation", {}).get("issues", [])

    # Start constructing the prompt
    prompt = ""
    prompt += "### Instructions\n"
    prompt += "Update the following source code based on the identified issues and fixes provided. "
    prompt += "Additionally, consider the context provided in `code_prompt` to ensure that the corrected source code adheres to the intended requirements.\n"
    prompt += "Only return the corrected source code enclosed in Python code fences.\n"
    prompt += "Ensure the corrected code fully adheres to the requirements specified in the document while maintaining the original structure as much as possible.\n"
    prompt += "\n"
    prompt += "#### Code Prompt (Context)\n"
    prompt += "This is the original prompt used to generate the initial source code:\n"
    prompt += "\n"
    prompt += f"{code_prompt}\n"
    prompt += "\n"
    prompt += "#### Original Source Code\n"
    prompt += "```python\n"
    prompt += f"{source_code}\n"
    prompt += "```\n"
    prompt += "\n"

    if issues:
        # Add issues and fixes if there are any
        prompt += "#### Issues\n"
        prompt += "Here are the identified issues in the source code:\n"
        prompt += "\n"
        for i, issue in enumerate(issues, 1):
            prompt += f"- Issue {i}: {issue.get('description', 'No detailed description provided.')}\n"
        prompt += "\n"
        prompt += "#### Fixes\n"
        prompt += "Here are the fixes for the identified issues:\n"
        prompt += "\n"
        for i, issue in enumerate(issues, 1):
            fix = issue.get("suggested_fix", "No suggested fix provided.")
            prompt += f"- Fix for Issue {i}: {fix}\n"
        prompt += "\n"

    # Add final instruction to return only the corrected code
    prompt += "### Final Output\n"
    prompt += "Return only the corrected source code in the following format:\n"
    prompt += "```python\n"
    prompt += "(Corrected source code here)\n"
    prompt += "```\n"


    # print("="*100)
    # print(prompt)
    # input("仕様書と比較してNoが出たモジュールの修正プロンプト")
    # print("="*100)

    response = model.generate_content(
        prompt,
        )
    
    code = clean_code_fence(response.text)
    

    # response = generate_code(prompt)

    # code = clean_code_fence(response)

    return code



def append_to_text(text, suffix):
    # 入力された文字列の末尾に、指定された文字列を追加する
    return text + suffix


def remove_text_based_on_marker(text, marker, option="before"):
    # マーカーの位置を取得
    index = text.find(marker)
    
    # マーカーが見つかった場合
    if index != -1:
        if option == "before":
            # マーカーより前を削除してマーカー以降を返す
            return text[index:].strip()
        elif option == "after":
            # マーカーを含み、それ以降をすべて削除
            return text[:index].strip()
        else:
            raise ValueError("Invalid option. Use 'before' or 'after'.")
    else:
        # マーカーが見つからなかった場合は元の文字列を返す
        return text






class ConcatenationError(Exception):
    """concat後のコードに実行エラーが多すぎた場合の例外"""
    pass





from config import NO_ERROR_CODE_DIR, CODE_PROMPT_DIR, TEMP_CODE_DIR

def main():
    directory_path = NO_ERROR_CODE_DIR
    generated_code_path = get_latest_file(directory_path)
    generated_code = read_file_content(generated_code_path)

    directory_path = CODE_PROMPT_DIR
    code_prompt_path = get_latest_file(directory_path)
    code_prompt = read_file_content(code_prompt_path)

    code_prompt_1 = remove_text_based_on_marker(code_prompt, "**1. Model Overview:**" ,"before")
    code_prompt_1 = remove_text_based_on_marker(code_prompt_1, "]," ,"after")
    suffix_1 = "\n    ]\n}\n```"
    code_prompt_1 = append_to_text(code_prompt_1, suffix_1)

    code_prompt_2 = remove_text_based_on_marker(code_prompt, "**2. Module Definition:**" ,"before")
    code_prompt_2 = remove_text_based_on_marker(code_prompt_2, "**4. Constraints and Dependencies:**" ,"after")

    code_prompt_eval = append_to_text(code_prompt_1+"\n", code_prompt_2)
    # print(code_prompt)
    # print("="*100)



    directory_path = os.path.join(CURRENT_DIR, "CONCATNATED_CODE")


    # 生成された【NO_ERROR_CODE】の評価
    evaluation = generated_code_evaluation(code_prompt_eval, generated_code)



    evaluation = extract_json_content(evaluation)

    print(evaluation)

    # print("%"*100)
    # print(generated_code_path)
    # input("^^このソースコードを評価！！")

    # print(type(evaluation))
    evaluation_json = json.loads(evaluation)
    # print(type(evaluation_json))
    eval = evaluation_json["evaluation"]["matches_specification"]

    # input("&&")

    # if eval == "No":
    # 生成されたコードが仕様書内容を満たさない場合
    count = 0
    while eval == "No":

        # 修正後のソースコード生成
        fixed_code = code_generation(generated_code, evaluation, code_prompt)

        # print("="*100)
        # print(fixed_code)

        # temp に保存
        filepath = save_code(fixed_code)


        result, errors = execute_generated_code(filepath)
        output = result + errors

        llmout = error_check(output)

        while True:        
            print(llmout)
            # input("="*100)

            # 実行時にエラー発生
            if llmout.strip() == "Error":

                temp_file_path = get_latest_file()

                # print("%"*100)
                # print("これがさっきのと同じならOK")
                # print(temp_file_path)
                # print("%"*100)

                # ファイル内容を読み取る
                try:
                    if os.path.exists(temp_file_path):
                        with open(temp_file_path, 'r', encoding='utf-8') as file:
                            code = file.read()
                            # print(f"ファイル '{temp_file_path}' の内容:\n")
                            # print(code)
                    else:
                        print(f"ファイル '{temp_file_path}' は存在しません。")
                except Exception as e:
                    print(f"ファイルを読み取る際にエラーが発生しました: {e}")

                specification_path = os.path.join(CURRENT_DIR, "CODE_PROMPT")
                specifications = get_latest_file(specification_path)

                advice = get_advice(code, errors, result)

                filepath = get_fixed_code(code, errors, result, None, advice)

                # print("%"*100)
                # print("新しく修正したコード")
                # print(filepath)
                # print("%"*100)

                result, errors = execute_generated_code(filepath)

                print("="*100)
                print("ERROR")
                print(errors)
                # input("="*100)


                llmout = error_check(result + errors)

                count += 1
                print("count = " + str(count) + "/3")
                if count == 3:
                    print("無理なのでbreak")
                    # エラーを明示的に発生させる
                    raise ConcatenationError("3回エラーが発生したため処理を中止します")
                


            # 実行時にエラー発生しない
            else:
                generated_code = read_file_content(filepath)
                evaluation = generated_code_evaluation(code_prompt_eval, generated_code)
                evaluation = extract_json_content(evaluation)

                evaluation_json = json.loads(evaluation)
                print("eval_fixed")
                print(evaluation)
                eval = evaluation_json["evaluation"]["matches_specification"]
                break


    else: # 修正点無し

        print("OK")
        print("\n")
        print("Next is concatnation.")


if __name__ == '__main__':
    main()
