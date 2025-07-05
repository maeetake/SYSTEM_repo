# 数値実験環境自動構築システム

このリポジトリは、自然言語の指示に基づき、数値実験環境を自動で構築するシステムを格納しています。
コード生成部分には、以下の3つの異なる大規模言語モデル（LLM）を利用したバージョンが含まれています。

- **AUTO_SYSTEM_GPT:** GPTを利用したシステム
- **AUTO_SYSTEM_Gemini:** Geminiを利用したシステム
- **AUTO_SYSTEM_Claude:** Claudeを利用したシステム

## 概要

本システムは、ユーザーが自然言語で実験の目的や手順を記述するだけで、必要なデータ処理、モデル構築、学習、評価、可視化までの一連のコードを自動で生成します。
これにより、プログラミングの詳細な知識がなくても、迅速に数値実験の環境を整えることが可能になります。

## システム構成

各`AUTO_SYSTEM_*`ディレクトリは、それぞれ独立したシステムとして機能し、以下の様な共通のファイル構成を持っています。

- `a_question_generator.py`: ユーザーの曖昧な指示から、具体的な質問を生成
- `b_QandA.py`: 生成された質問とユーザーの回答を構造化
- `c_model_specification_author.py`: 構造化された情報から、モデルの仕様書を作成
- `d_specification_analyst.py`: 仕様書を分析し、タスクを分解
- `e_module_and_interface_definer.py`: 分解されたタスクから、モジュールとインターフェースを定義
- `f_prompt_generator.py`: 定義されたモジュールに基づき、コード生成用のプロンプトを作成
- `g_code_generator.py`: プロンプトをLLMに渡し、コードを生成
- `h_code_executor.py`: 生成されたコードを実行
- `i_code_error_corrector.py`: コード実行時に発生したエラーを修正
- `j_code_checker.py`: コードの品質をチェック
- `k_code_concatenator.py`: 生成されたコードを結合
- `l_save_output.py`: 最終的な出力を保存
- `main.py`, `main_2.py`: システムの実行スクリプト
- `config.py`: 各種設定ファイル
- `*_client.py`: 各LLMのAPIクライアント

## セットアップ

1. リポジトリをクローンします。
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   ```
2. 必要なライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
3. 各LLMのAPIキーを環境変数に設定します。これにより、コード内に直接キーを記述する必要がなくなり、セキュリティが向上します。

   **Windows (コマンドプロンプト)**
   ```bash
   setx OPENAI_API_KEY "YOUR_OPENAI_API_KEY"
   setx GEMINI_API_KEY "YOUR_GEMINI_API_KEY"
   setx CLAUDE_API_KEY "YOUR_CLAUDE_API_KEY"
   ```
   コマンドプロンプトを再起動すると、環境変数が反映されます。

   **Windows (PowerShell)**
   ```powershell
   $env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   $env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   $env:CLAUDE_API_KEY="YOUR_CLAUDE_API_KEY"
   ```
   恒久的に設定する場合は、`[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY", "User")`のように実行してください。

   **macOS / Linux**
   ```bash
   export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   export CLAUDE_API_KEY="YOUR_CLAUDE_API_KEY"
   ```
   この設定を永続化させるには、`.bashrc`や`.zshrc`などのシェル設定ファイルに上記の行を追記してください。

   **注意:** `config.py`内のAPIキー読み込み処理が、これらの環境変数を参照するように実装されている必要があります。

## 使い方

各`AUTO_SYSTEM_*`ディレクトリに移動し、`main.py`または`main_2.py`を実行します。

- **`main.py`**: 仕様書作成部分を実行します。
- **`main_2.py`**: コード生成部分を実行します。

```bash
cd AUTO_SYSTEM_GPT  # or AUTO_SYSTEM_Gemini, AUTO_SYSTEM_Claude

# 仕様書作成
python main.py

# コード生成
python main_2.py
```
