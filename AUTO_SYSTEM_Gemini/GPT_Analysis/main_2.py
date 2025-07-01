import time
import f_prompt_generator as prompt_gen
import g_code_generator as code_gen
import i_code_error_corrector as error_corrector
import j_code_checker as code_checker
import k_code_concatenator as code_concat
import l_save_output as save_output

import subprocess
import os
import sys
from config import PROJECT_ROOT

first_prompt = os.path.join(PROJECT_ROOT, "f_prompt_generator.py")
make_code = os.path.join(PROJECT_ROOT, "g_code_generator.py")

python_interpreter = sys.executable

def main():
    # 実行時間の計測開始
    start_time = time.time()
    iteration_times = []  # 各イテレーションごとの実行時間を格納するリスト

    print("=== 全体プロセスの開始 ===")

    num = 8

    for iteration in range(num):
        print(f"\n=== 実行サイクル {iteration + 1}/{num} ==="+"*"*100)
        
        # イテレーション開始時間を取得
        start_iteration_time = time.time()

        if iteration == 0:
            print("\n=== ステップ 1: プロンプト生成 (初回) ===")
            process = subprocess.run(
                [python_interpreter, first_prompt],
                input=f"{iteration + 1}\nno\n",
                text=True,
            )
            print("初回のプロンプト生成が完了しました。\n")
        else:
            print("\n=== ステップ 1: プロンプト生成 (2回目以降) ===")
            process = subprocess.run(
                [python_interpreter, first_prompt],
                input=f"{iteration + 1}\nyes\n",
                text=True,
            )
            print("2回目以降のプロンプト生成が完了しました。\n")


        while True:
            try:
                print("=== ステップ 2: コード生成 ===")
                process = subprocess.run(
                    [python_interpreter, make_code],
                    input=str(iteration + 1) + "\n",
                    text=True,
                )
                print("コード生成が完了しました。\n")

                print("=== ステップ 3: エラー修正 ===")
                error_corrector.main()
                break

            except Exception as e:
                print(f"エラー発生: {e}")
                print("再試行中")
                time.sleep(2)
        print("エラー修正が完了しました。\n")
                

        # code_checker.main()
        while True:
            try:
                print("=== ステップ 4: コードのチェック ===")
                code_checker.main()
                break  # 正常終了した場合はループを抜ける
            except Exception as e:
                print(f"エラー発生: {e}")
                print("再試行中")
                time.sleep(2)
        print("コードのチェックが完了しました。\n")


        
        while True:
            try:
                print("=== ステップ 5: コードの統合 ===")
                code_concat.main()
                break  # 正常終了した場合はループを抜ける
            except Exception as e:
                print(f"エラー発生: {e}")
                print("再試行中")
                time.sleep(2)
        print("コードの統合が完了しました。\n")

        while True:
            try:
                print("=== ステップ 6: 実行結果の保存 ===")
                save_output.main()
                break  # 正常終了した場合はループを抜ける
            except Exception as e:
                print(f"エラー発生: {e}")
                print("再試行中")
                time.sleep(2)
        print("実行結果の保存が完了しました。\n")

        # イテレーション終了時間を取得し、差分を計算
        end_iteration_time = time.time()
        iteration_elapsed_time = end_iteration_time - start_iteration_time
        iteration_times.append(f"Iteration {iteration + 1}: {iteration_elapsed_time:.2f} 秒")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n=== 実行にかかった時間: {elapsed_time:.2f} 秒 ===")

    print("=== 全てのプロセスが正常に完了しました。 ===")

    # 実行時間の計測終了
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n=== 全体の実行にかかった時間: {elapsed_time:.2f} 秒 ===")

    # イテレーションごとの時間をファイルに保存
    with open("iteration_times.txt", "w", encoding="utf-8") as file:
        file.write("各イテレーションごとの実行時間:\n")
        file.write("\n".join(iteration_times))
        file.write(f"\n\n全体の実行時間: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()