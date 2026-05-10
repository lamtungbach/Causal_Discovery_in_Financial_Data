import os
import zipfile
from collections import defaultdict
from pathlib import Path

# ===== CONFIG =====
PROJECT_ROOT = "."          # thư mục project hiện tại
OUTPUT_DIR = "csv_exports"  # nơi chứa zip
MIN_SIZE_MB = 1             # csv lớn hơn mức này sẽ được log
# ==================

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_groups = defaultdict(list)

print("\n🔍 Scanning project...\n")

# Scan toàn bộ project
for root, dirs, files in os.walk(PROJECT_ROOT):

    # bỏ các folder không cần
    ignore_dirs = [".git", "venv", "__pycache__", "node_modules"]

    dirs[:] = [d for d in dirs if d not in ignore_dirs]

    for file in files:
        if file.lower().endswith(".csv"):

            full_path = os.path.join(root, file)

            # size MB
            size_mb = os.path.getsize(full_path) / (1024 * 1024)

            # group theo folder
            relative_folder = os.path.relpath(root, PROJECT_ROOT)

            csv_groups[relative_folder].append({
                "path": full_path,
                "size_mb": size_mb
            })

# ===== Report =====

print("=" * 60)
print("📁 CSV GROUP REPORT")
print("=" * 60)

for folder, files in csv_groups.items():

    total_size = sum(f["size_mb"] for f in files)

    print(f"\n📂 Folder: {folder}")
    print(f"   Total CSVs: {len(files)}")
    print(f"   Total Size: {total_size:.2f} MB")

    for f in files:

        large_tag = "🔥 LARGE" if f["size_mb"] >= MIN_SIZE_MB else ""

        print(
            f"   - {Path(f['path']).name} "
            f"({f['size_mb']:.2f} MB) {large_tag}"
        )

# ===== ZIP =====

print("\n📦 Creating ZIP groups...\n")

for folder, files in csv_groups.items():

    # tên zip an toàn
    safe_folder_name = folder.replace("\\", "_").replace("/", "_")

    if safe_folder_name == ".":
        safe_folder_name = "root"

    zip_name = f"{safe_folder_name}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

        for f in files:

            arcname = os.path.relpath(f["path"], PROJECT_ROOT)

            zipf.write(f["path"], arcname)

    print(f"✅ Created: {zip_path}")

print("\n🎉 Done!")