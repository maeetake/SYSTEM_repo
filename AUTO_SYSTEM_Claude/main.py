import a_question_generator as question_gen
import b_QandA as q_and_a
import c_model_specification_author as model_spec_author
import d_specification_analyst as spec_analyst
import e_module_and_interface_definer as module_definer


def main():
    print("=== ステップ 1: 質問生成 ===")
    question_gen.main()
    print("質問生成が完了しました。\n")

    print("=== ステップ 2: 回答の構造化と要約 ===")
    q_and_a.main()
    print("回答の構造化と要約が完了しました。\n")

    print("=== ステップ 3: モデル仕様の生成 ===")
    model_spec_author.main()
    print("モデル仕様の生成が完了しました。\n")

    print("=== ステップ 4: 仕様書の分析とタスク分解 ===")
    spec_analyst.main()
    print("仕様書の分析とタスク分解が完了しました。\n")

    print("=== ステップ 5: モジュール定義とインターフェースの生成 ===")
    module_definer.main()
    print("モジュール定義とインターフェースの生成が完了しました。\n")

    print("=== 全てのプロセスが正常に完了しました。 ===")


if __name__ == "__main__":
    main()



"""
1. What specific data sources do you intend to use for training and evaluating the model (e.g., historical stock prices, financial news, social media sentiment)?
Your answer: For training and evaluation, I will exclusively use the CSV file containing daily OHLC data provided by the user. No additional data sources such as financial news or social media sentiment will be used.
2. What is the desired time granularity of the prediction (e.g., daily, hourly)?
Your answer: The prediction will be made on a daily basis, forecasting the closing price for the next day.
3. What specific time frame of historical data are you planning to utilize for model training and backtesting?
Your answer: I will use all of the historical daily OHLC data available in the provided CSV file for training and backtesting. The specific time frame depends on the data in the file, but the entire available period will be utilized.
4. What is the target variable for prediction? Is it the closing price, the highest price, the lowest price, or some other metric related to NVIDIA's stock?
Your answer: The target variable for prediction is the closing price one day ahead.
5. What evaluation metrics will be used to assess the performance of the prediction models (e.g., Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), accuracy)?
Your answer: I will assess the model performance using regression error metrics, specifically Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
6. What machine learning models are you considering for prediction (e.g., linear regression, time series models, neural networks)?
Your answer: I plan to use LSTM neural network to capture non-linear temporal dependencies.
7. What features or input variables do you plan to utilize alongside the historical stock price for building the predictive model?
Your answer: The primary input variables will be the provided OHLC data. In addition, I will engineer supplementary features such as moving averages, and other technical indicators derived from the OHLC values.
8. What programming language and libraries are you planning on using for model building and data analysis (e.g., Python with scikit-learn, TensorFlow, PyTorch)?
Your answer: I will use Python for model building and data analysis. For data processing, I will utilize libraries such as Pandas and NumPy, employ scikit-learn for basic machine learning models, and use TensorFlow for implementing neural networks.
9. Are there any specific visualization requirements for the prediction results (e.g., line graphs, charts displaying predicted versus actual prices)?
Your answer: Yes, the visualization will include line charts that clearly display and compare the predicted closing prices with the actual closing prices.
10. What data preprocessing steps do you anticipate taking for handling noisy or missing data in the input datasets?        
Your answer: I will begin by detecting any missing values or outliers in the CSV data. Depending on the situation, missing data will be imputed or removed. Additionally, I will perform normalization or scaling on the features as necessary and address outliers using techniques such as clipping or smoothing.
"""