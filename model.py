import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from config import data_base_path
import random
import requests
import retrying

forecast_price = {}

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 10000  # Giới hạn số lượng dữ liệu tối đa khi lưu trữ
INITIAL_FETCH_SIZE = 10000  # Số lượng nến lần đầu tải về

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, interval="1m", limit=1000, start_time=None, end_time=None):
    try:
        base_url = "https://fapi.binance.com"
        endpoint = f"/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        url = base_url + endpoint
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f'Failed to fetch prices for {symbol} from Binance API: {str(e)}')
        raise e

def add_wyckoff_features(df):
    # Thêm chỉ số Accumulation/Distribution (A/D)
    df['AD'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']

    # Thêm chỉ số Price-Volume Trend (PVT)
    df['PVT'] = (df['close'] - df['close'].shift(1)) * df['volume']
    df['PVT'] = df['PVT'].cumsum()  # Tính tổng tích lũy

    # Thêm chỉ số Chaikin Money Flow (CMF)
    df['CMF'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['CMF'] = df['CMF'].rolling(window=20).sum()  # Trung bình động trong 20 kỳ

    # Loại bỏ các giá trị NaN sau khi tính toán các chỉ số
    df = df.dropna()

    return df

def download_data(token):
    symbols = f"{token.upper()}USDT"
    interval = "1m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    
    # Đường dẫn file CSV để lưu trữ
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")

    # Kiểm tra xem file có tồn tại hay không
    if os.path.exists(file_path):
        # Tính thời gian bắt đầu cho 100 cây nến 5 phút
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 100, start_time, end_time)
    else:
        # Nếu file không tồn tại, tải về số lượng INITIAL_FETCH_SIZE nến
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE*5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)

    # Chuyển dữ liệu thành DataFrame
    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Kiểm tra và đọc dữ liệu cũ nếu tồn tại
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        # Kết hợp dữ liệu cũ và mới
        combined_df = pd.concat([old_df, new_df])
        # Loại bỏ các bản ghi trùng lặp dựa trên 'start_time'
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    # Giới hạn số lượng dữ liệu tối đa
    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    # Lưu dữ liệu đã kết hợp vào file CSV
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_1m_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)

    # Sử dụng các cột sau (đúng với dữ liệu bạn đã lưu)
    columns_to_use = [
        "start_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]

    # Kiểm tra nếu tất cả các cột cần thiết tồn tại trong DataFrame
    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "open", "high", "low", "close", "volume",
            "end_time", "quote_asset_volume", "n_trades", 
            "taker_volume", "taker_volume_usd"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')
        df.index.name = "date"

        # Thêm các chỉ số Wyckoff vào dữ liệu
        df = add_wyckoff_features(df)

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        print(f"Formatted data with Wyckoff features saved to {output_path}")
    else:
        print(f"Required columns are missing in {file_path}. Skipping this file.")

def train_model(token):
    # Hiển thị thời gian dự đoán hiện tại
    time_start = datetime.now()

    # Load the token price data
    price_data = pd.read_csv(os.path.join(data_base_path, f"{token.lower()}_price_data.csv"))
    df = pd.DataFrame()

    # Convert 'date' to datetime
    price_data["date"] = pd.to_datetime(price_data["date"])

    # Set the date column as the index for resampling
    price_data.set_index("date", inplace=True)

    # Resample the data to 1-minute frequency and compute the mean price
    df = price_data.resample('1T').mean()

    # Prepare data for Linear Regression
    df = df.dropna()  # Loại bỏ các giá trị NaN (nếu có)
    X = np.array(df[['close', 'AD', 'PVT', 'CMF']]).reshape(-1, 4)  # Sử dụng các đặc trưng bao gồm các chỉ số Wyckoff
    y = df['close'].values  # Sử dụng giá đóng cửa làm mục tiêu

    # Khởi tạo mô hình Linear Regression
    model = LinearRegression()
    model.fit(X, y)  # Huấn luyện mô hình

    # Dự đoán giá tiếp theo
    next_time_index = np.array([[df['close'].iloc[-1], df['AD'].iloc[-1], df['PVT'].iloc[-1], df['CMF'].iloc[-1]]])  # Giá trị thời gian tiếp theo
    predicted_price = model.predict(next_time_index)[0]  # Dự đoán giá

    # Xác định khoảng dao động xung quanh giá dự đoán
    fluctuation_range = 0.005 * predicted_price  # Lấy 1% của giá dự đoán làm khoảng dao động
    min_price = predicted_price - fluctuation_range
    max_price = predicted_price + fluctuation_range

    # Chọn ngẫu nhiên một giá trị trong khoảng dao động
    price_predict = random.uniform(min_price, max_price)
    forecast_price[token] = price_predict

    print(f"Forecasted price for {token}: {forecast_price[token]}")

    time_end = datetime.now()
    print(f"Time elapsed forecast: {time_end - time_start}")

def update_data():
    tokens = ["ETH", "BTC", "BNB", "SOL", "ARB"]
    for token in tokens:
        download_data(token)
        format_data(token)
        train_model(token)

if __name__ == "__main__":
    update_data()
