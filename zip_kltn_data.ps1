# zip_kltn_data.ps1
# Zip các file CSV của project KLTN theo nhóm folder
# Bỏ qua .venv và venv (thư viện)
# Chạy: .\zip_kltn_data.ps1

$ROOT     = "D:\KLTN"
$OUT_DIR  = "D:\KLTN_zip"

# Tạo thư mục output
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
Write-Host "`n📦 Bắt đầu zip CSV theo nhóm..." -ForegroundColor Cyan

# ─────────────────────────────────────────────────────────────
# Nhóm 1: data_raw  ←  data\raw\
# ─────────────────────────────────────────────────────────────
$group1_files = @(
    "$ROOT\data\raw\raw_prices.csv",
    "$ROOT\data\raw\vnindex_manual.csv"
)
$zip1 = "$OUT_DIR\KLTN_data_raw.zip"
if (Test-Path $zip1) { Remove-Item $zip1 }
Compress-Archive -Path $group1_files -DestinationPath $zip1
$size1 = [math]::Round((Get-Item $zip1).Length / 1KB, 1)
Write-Host "  ✅ KLTN_data_raw.zip       ($size1 KB)  ← data\raw\" -ForegroundColor Green

# ─────────────────────────────────────────────────────────────
# Nhóm 2: data_processed  ←  data\processed\ + data\results\
# ─────────────────────────────────────────────────────────────
$group2_files = @(
    "$ROOT\data\processed\log_returns.csv",
    "$ROOT\data\processed\merged_prices.csv",
    "$ROOT\data\processed\prices_clean.csv",
    "$ROOT\data\processed\volatility_21d.csv",
    "$ROOT\data\processed\volatility_stationary.csv",
    "$ROOT\data\results\adf_kpss_results.csv",
    "$ROOT\data\results\correlation_matrix.csv",
    "$ROOT\data\results\summary_stats.csv"
)
# Lọc chỉ file tồn tại
$group2_exist = $group2_files | Where-Object { Test-Path $_ }
$zip2 = "$OUT_DIR\KLTN_data_processed.zip"
if (Test-Path $zip2) { Remove-Item $zip2 }
Compress-Archive -Path $group2_exist -DestinationPath $zip2
$size2 = [math]::Round((Get-Item $zip2).Length / 1KB, 1)
Write-Host "  ✅ KLTN_data_processed.zip ($size2 KB)  ← data\processed\ + data\results\" -ForegroundColor Green

# ─────────────────────────────────────────────────────────────
# Nhóm 3: reports  ←  reports\ + reports\tables\
# ─────────────────────────────────────────────────────────────
$group3_files = @(
    "$ROOT\reports\adj_NeuralGC_cLSTM.csv",
    "$ROOT\reports\adj_NeuralGC_cMLP.csv",
    "$ROOT\reports\adj_NeuralGC_Ensemble.csv",
    "$ROOT\reports\adj_NOTEARS.csv",
    "$ROOT\reports\adj_PC.csv",
    "$ROOT\reports\tables\hub_evolution_neural.csv",
    "$ROOT\reports\tables\subperiod_results_neural.csv",
    "$ROOT\reports\tables\subperiod_shd_matrix_neural.csv"
)
$group3_exist = $group3_files | Where-Object { Test-Path $_ }
$zip3 = "$OUT_DIR\KLTN_reports.zip"
if (Test-Path $zip3) { Remove-Item $zip3 }
Compress-Archive -Path $group3_exist -DestinationPath $zip3
$size3 = [math]::Round((Get-Item $zip3).Length / 1KB, 1)
Write-Host "  ✅ KLTN_reports.zip        ($size3 KB)  ← reports\ + reports\tables\" -ForegroundColor Green

# ─────────────────────────────────────────────────────────────
# Tổng kết
# ─────────────────────────────────────────────────────────────
Write-Host "`n══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  📂 Output: $OUT_DIR" -ForegroundColor Cyan
Write-Host "  3 file zip sẵn sàng upload Google Drive:" -ForegroundColor Cyan
Get-ChildItem $OUT_DIR -Filter "*.zip" | ForEach-Object {
    $mb = [math]::Round($_.Length / 1KB, 1)
    Write-Host "    📦 $($_.Name)  ($mb KB)" -ForegroundColor Yellow
}
Write-Host "══════════════════════════════════════════`n" -ForegroundColor Cyan