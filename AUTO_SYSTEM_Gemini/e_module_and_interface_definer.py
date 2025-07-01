"""
分解したタスクの詳細設定
"""
import json
import google.generativeai as genai
import os
import typing_extensions as typing
import time
from b_QandA import get_json, save_to_json_file, extract_json_content

# Google Gemini APIキーを設定
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

from config import MODEL_NAME

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

class constraints_schema(typing.TypedDict):
    library_versions_and_configurations: str
    error_handling: str
    input_formats_and_data_types: str
    output_formats_and_data_types: str
    specific_error_handling: str

class task_schema(typing.TypedDict):
    task_name: str
    model_role_and_purpose: str
    concrete_tasks: str
    dependencies: list[str]
    constraints: constraints_schema
    code_skeleton: str
    documentation: str



GENERATION_CONFIG = genai.GenerationConfig(
        temperature=1,
        top_p=1,
        top_k=5,
        response_mime_type="application/json",
        # response_schema=task_schema
)



def retry_on_json_decode_error(func):
    """
    Decorator to retry the function in case of a JSONDecodeError.
    """
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except json.JSONDecodeError:
                print(f"JSON decode error occurred. Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2)  # Wait before retrying
        print("Max retries reached. Exiting.")
        return None
    return wrapper


@retry_on_json_decode_error
def generate_module_definition(module_name, generate_task, model_specification):
    """
    仕様書と分解されたタスクからモジュールの定義とインターフェースを生成。

    Args:
      module_name: 生成するモジュール名
      generate_task: タスク情報
      model_specification: モデルの仕様書

    Returns:
      dict: モジュール定義とインターフェース情報
    """

    model_specification = json.loads(model_specification)

    prompt = ""
    prompt += "You are an AI assistant that defines a module: **" + module_name + "** for a source code generation system.\n"
    prompt += "Given the overview of the entire system, name of the module which you defines and the task it is designed to perform,\n"
    prompt += "please generate a detailed module definition, including its interface.\n\n"
    prompt += "Here's the overall task structure:\n"
    prompt += f"{json.dumps(model_specification, indent=4)}"
    prompt += "\n\n"
    prompt += "Now, focus on the module" + module_name + "**only**.\n"
    
    prompt += "This module will perform the task described as:\n"
    prompt += f"{json.dumps(generate_task, indent=4)}\n\n"

    prompt += "## Considering the overall context, please provide the module definition in the following JSON format, conforming to the provided schema.\n\n"
    prompt += "For each module, generate a complete JSON object with all the following fields:\n\n"

    prompt += "1.  **task_name:** A unique name for the task.\n"
    prompt += "2.  **module_responsibility:** Define the scope and responsibility of this module to ensure it focuses on a unique aspect of the overall pipeline without overlapping tasks.\n"
    prompt += "3.  **model_role_and_purpose:** Clearly describe the module's function and its purpose within the overall system. Avoid covering responsibilities handled by other modules.\n"
    prompt += "4.  **concrete_tasks:** Describe in detail the specific processing steps the module should execute. Each step must be unique to this module and should not duplicate tasks found in others. Decompose them into functions, classes, or modules, and specify the types and formats of inputs and outputs.\n"
    prompt += "5.  **dependencies:** Specify the execution order of each task and any dependencies on outputs from other modules as a list of task_names. Ensure dependencies do not create circular references.\n"
    prompt += "6. **constraints:** Detailed constraints for each task:\n"
    prompt += "    *   **library_versions_and_configurations:** Specify the versions and configurations of the libraries and frameworks to be used.\n"
    prompt += "    *   **error_handling:** Describe how exceptions and errors should be handled within the module. Clarify any boundary cases where errors should propagate to other modules.\n"
    prompt += "    *   **input_formats_and_data_types:** Describe the expected input formats and data types unique to this module.\n"
    prompt += "    *   **output_formats_and_data_types:** Describe the output formats and data types that downstream modules should consume. Ensure the outputs are distinct and necessary for dependencies.\n"
    prompt += "7.  **code_skeleton (if possible):** Provide code skeletons (e.g., function or class definitions) corresponding to each task, tailored to the programming language to be used. Clearly define interfaces to avoid confusion about task ownership.\n"
    prompt += "8.  **documentation:** Describe the purpose, implementation methods, and usage of each task. Clearly state the boundaries of what the module will and will not do, emphasizing collaboration with other modules.\n\n"

    prompt += "## The output must follow the following JSON format. Ensure that the output has the attributes mentioned above, and the constraints object is nested correctly.\n"
    prompt += "```json\n"
    prompt += "{\n"
    prompt += "  \"task_name\": \"Name of the task\",\n"
    prompt += "  \"module_responsibility\": \"Clearly define the module's responsibility to avoid task duplication\",\n"
    prompt += "  \"model_role_and_purpose\": \"Clearly describe the module's function and purpose within the system\",\n"
    prompt += "  \"concrete_tasks\": \"Detailed steps of the task, decomposed into functions, classes, or modules with input/output types and formats\",\n"
    prompt += "  \"dependencies\": [\"Dependent task name 1\", \"Dependent task name 2\", ...],\n"
    prompt += "  \"constraints\": {\n"
    prompt += "    \"library_versions_and_configurations\": \"Specify versions and configurations of libraries/frameworks\",\n"
    prompt += "    \"error_handling\": \"Describe how exceptions and errors should be handled\",\n"
    prompt += "    \"input_formats_and_data_types\": \"Describe input format and data type\",\n"
    prompt += "    \"output_formats_and_data_types\": \"Describe output format and data type\",\n"
    prompt += "    \"specific_error_handling\": \"Describe handling of specific errors (logging, notifications, etc.)\"\n"
    prompt += "  },\n"
    prompt += "  \"code_skeleton\": \"Provide code skeletons (e.g., function or class definitions) with clear interfaces\",\n"
    prompt += "  \"documentation\": \"Describe the purpose, implementation methods, and usage of the task\"\n"
    prompt += "}\n"
    prompt += "```"



    # print(prompt)
    # input("="*100)

    response = model.generate_content(
        prompt,
        generation_config=GENERATION_CONFIG
        )

    try:
        # print("="*100)
        # print(response.text)
        # print(type(response.text))
        # input("="*100)
        out_json = extract_json_content(response.text)
        # print(out_json)
        # input("="*100)


        save_to_json_file(json.dumps(out_json, indent=4, ensure_ascii=False), "5____"+module_name+".json")

        task_decomposition = json.loads(out_json)

    except json.JSONDecodeError:
        print("JSON decode error occurred.")
        print("response:")
        print(response.text)
        return None

    return task_decomposition



def define_all_module(task_decomposition, model_specification):
    """
    analyze_specification を全てのモジュールに対して実行
    """

    modules = {}

    all_tasks = task_decomposition["tasks"]

    for task in enumerate(all_tasks):

        generate_task = task[1]

        module_name = task[1]["task_name"].lower().replace(" ", "_")

        print(module_name)
        # input("="*100)
        module_definition = generate_module_definition(module_name, generate_task, model_specification)

        if module_definition:
            modules[module_name] = module_definition
        else:
            print(f"Failed to generate module definition for: {module_name}")
            return None

    return modules


def main():

    filename = "3.model_specification.json"
    model_specification = get_json(filename)
    # print(type(model_specification))
    model_specification = json.dumps(model_specification)


    filename = "4.task_decomposition.json"
    task_decomposition = get_json(filename)

    # print(task_decomposition)

    defined_modules = define_all_module(task_decomposition, model_specification)

    defined_modules = json.dumps(defined_modules, indent=4, ensure_ascii=False)

    if defined_modules:
        filename = "5.defined_modules.json"
        save_to_json_file(defined_modules, filename)
    else:
        print("Failed to decompose tasks.")


if __name__ == '__main__':
    main()