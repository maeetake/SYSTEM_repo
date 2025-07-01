import os
from datetime import datetime, timedelta
import csv
import numpy as np
import matplotlib.pyplot as plt

def generate_random_slope_changes_with_random_timing(x_range, num_segments, slope_values, sin_coefficient=0.01):
    """
    傾きが各セグメント内のランダムなタイミングで変化する y 配列を生成する関数。

    パラメータ:
        x_range (tuple): x の値の範囲を (開始, 終了) で指定。
        num_segments (int): 傾きが変化する区間の数。
        slope_values (list or numpy.ndarray): ランダムに選択する可能性のある傾きの値のリスト。
        sin_coefficient (float): sin(x) 項の係数。

    戻り値:
        x (numpy.ndarray): x の値の配列。
        y (numpy.ndarray): 対応する y の値の配列。
    """
    # x の値を生成
    x = np.linspace(x_range[0], x_range[1], 50000)
    segment_length = len(x) // num_segments  # 各セグメントの長さを計算

    # ランダムな傾きのリストを生成
    slopes = np.random.choice(slope_values, size=num_segments)

    # y の初期化
    y = np.zeros_like(x)
    current_y = 0  # 現在の y の開始値

    # 各セグメントごとにランダムなタイミングで傾きを変更
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_segments - 1 else len(x)

        # セグメント内のランダムな位置を選択（傾きが変わるタイミング）
        if end - start > 1:  # セグメント内に少なくとも2つのデータポイントが必要
            change_point = np.random.randint(start + 1, end)
        else:
            change_point = end

        # # デバッグ用: セグメントの範囲と傾きが変わる位置を出力
        # print(f"セグメント {i+1} の範囲: start={start}, end={end}, change_point={change_point}, 傾き={slopes[i]:.3f}")

        # セグメントの前半部分の計算（常に slopes[i] を用いる）
        y[start:change_point] = current_y + slopes[i] * (x[start:change_point] - x[start]) \
                                 + sin_coefficient * np.sin(x[start:change_point])
        current_y = y[change_point - 1]  # 傾きが変わる位置の y 値を更新

        # 後半部分の計算
        if i + 1 < num_segments:
            # 最後のセグメント以外は次の傾き (slopes[i+1]) を用いる
            next_slope = slopes[i + 1]
            y[change_point:end] = current_y + next_slope * (x[change_point:end] - x[change_point]) \
                                   + sin_coefficient * np.sin(x[change_point:end])
            current_y = y[end - 1]  # 次のセグメント用に y を更新
        else:
            # 最後のセグメントでは slopes[i] を用いて計算
            y[change_point:end] = current_y + slopes[i] * (x[change_point:end] - x[change_point]) \
                                   + sin_coefficient * np.sin(x[change_point:end])
            current_y = y[end - 1]

    return x, y


def generate_ohlc_data(x, y, num_segments=500):
    """
    y 配列から OHLC（Open, High, Low, Close）データを生成する関数。

    パラメータ:
        x (numpy.ndarray): x の値の配列。
        y (numpy.ndarray): y の値の配列。
        num_segments (int): データを分割するセグメントの数。

    戻り値:
        ohlc (list of tuples): 各セグメントごとの OHLC タプルのリスト。
    """
    segment_length = len(x) // num_segments  # 各セグメントの長さを計算
    ohlc = []

    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_segments - 1 else len(x)
        segment = y[start:end]
        open_price = segment[0]  # 開始時の価格
        high_price = np.max(segment)  # 最高価格
        low_price = np.min(segment)  # 最低価格
        close_price = segment[-1]  # 終了時の価格
        ohlc.append((open_price, high_price, low_price, close_price))  # OHLC データを追加

    return ohlc


def save_ohlc_to_csv(ohlc_data, filename):
    """
    OHLC データを CSV ファイルに保存する関数（日時情報を含む）。

    パラメータ:
        ohlc_data (list of tuples): 保存する OHLC データ。
        filename (str): ファイルの保存先パス。
    """
    # ディレクトリが存在しない場合は作成
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(CURRENT_DIR, "UNITTEST_DATA", filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 各行に対応する日時を生成
    start_date = datetime.now()
    dates = [start_date + timedelta(days=i) for i in range(len(ohlc_data))]

    # CSV ファイルにデータを書き込み
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Open", "High", "Low", "Close"])  # ヘッダー行
        for date, ohlc in zip(dates, ohlc_data):
            writer.writerow([date.strftime("%Y-%m-%d"), *ohlc])

    print(f"日時付きの OHLC データを {file_path} に保存しました")


# テスト用のパラメータ設定
x_range = (0, 5000)
num_segments = 1000
slope_values = np.arange(-0.02, 0.03, 0.01)  # ランダムな傾きの候補
sin_coefficient = 0.01  # 正弦波の影響を調整

# ランダムなタイミングで傾きが変化するデータを生成
x, y = generate_random_slope_changes_with_random_timing(x_range, num_segments, slope_values, sin_coefficient)

ohlc_data = generate_ohlc_data(x, y)

save_ohlc_to_csv(ohlc_data, "0.data_for_unittest.csv")

# OHLC データを出力
print("OHLC データ:")
for i, (o, h, l, c) in enumerate(ohlc_data):
    print(f"セグメント {i + 1}: Open={o:.2f}, High={h:.2f}, Low={l:.2f}, Close={c:.2f}")

# グラフの描画
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='y', color='orange')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # x 軸
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # y 軸
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
