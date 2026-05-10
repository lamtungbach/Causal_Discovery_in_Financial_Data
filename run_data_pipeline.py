"""
run_data_pipeline.py — Entry point chính.
Đã sửa lỗi gọi Class DataPreprocessor.
"""
from src.data.downloader      import download_all
from src.data.preprocessor    import DataPreprocessor
from src.data.validator       import run_validation
from src.visualization.plotter import plot_volatility, plot_correlation

def main():
    print("=" * 60)
    print("   PIPELINE THU THẬP DỮ LIỆU — NOTEARS THESIS")
    print("=" * 60)

    # Bước 1: Download - Hàm này trả về DataFrame prices sạch sẽ
    prices = download_all() 

    # Bước 2: Xử lý
    # Khởi tạo Preprocessor với window=21 (khớp với nghiên cứu của bạn)
    prep = DataPreprocessor(window=21)
    
    # QUAN TRỌNG: Dùng fit_transform và truyền 'prices' từ Bước 1 vào
    # Điều này giúp tận dụng dữ liệu vừa tải và không bị chạy fetch_prices() bên trong lần nữa
    X, vol = prep.fit_transform(prices)

    # Bước 3: Kiểm định
    # Bạn có thể dùng hàm test_stationarity đã có trong prep
    print("\n🔍 Đang kiểm định tính dừng (ADF/KPSS)...")
    val_results = prep.test_stationarity(vol)
    print(val_results)

    # Bước 4: Biểu đồ
    print("\n🎨 Đang vẽ biểu đồ...")
    plot_volatility(vol)
    plot_correlation(vol)

    # Lưu dữ liệu cuối cùng để làm Input cho NOTEARS
    vol.to_csv("data/processed/volatility_21d.csv")

    print("\n" + "=" * 60)
    print(f"   HOÀN THÀNH! Dữ liệu sẵn sàng: {X.shape}")
    print("   File: data/processed/volatility_21d.csv")
    print("=" * 60)