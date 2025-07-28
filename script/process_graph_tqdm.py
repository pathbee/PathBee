import os
import gzip
import shutil
import urllib.request
from tqdm import tqdm
import zipfile

# === 配置 ===
url = "https://nrvis.com/download/data/dimacs10/inf-europe_osm.zip"
compressed_filename = os.path.basename(url)
expected_txt_name = "inf-europe_osm.mtx"
processed_filename = "inf-europe_osm-processed.txt"

raw_dir = "data/raw"
processed_dir = "data/processed"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

compressed_path = os.path.join(raw_dir, compressed_filename)
txt_path = os.path.join(raw_dir, expected_txt_name)
output_path = os.path.join(processed_dir, processed_filename)

# === 下载压缩包 ===
if not os.path.exists(compressed_path):
    print(f"Downloading from {url} ...")
    urllib.request.urlretrieve(url, compressed_path)
    print("Download complete.")

# === 解压 zip 或 gz ===
if not os.path.exists(txt_path):
    print(f"Extracting {compressed_path} ...")
    if compressed_path.endswith(".zip"):
        with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
            print("Extracted files:", zip_ref.namelist())
    elif compressed_path.endswith(".gz"):
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(txt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError("Unsupported compressed format.")
    print("Extraction complete.")

# === 判断是否是 Matrix Market 格式 ===
is_mtx = False
with open(txt_path, 'r') as f:
    for line in f:
        if line.startswith("%%MatrixMarket"):
            is_mtx = True
        break  # 只看第一行就够

# === 计算总边数（仅用于进度条） ===
print("Counting total lines for progress bar...")
with open(txt_path, "r") as f:
    total_lines = sum(
        1 for line in f
        if line.strip() and not line.startswith('%') and not line.startswith('#') and len(line.split()) >= 2
    )
print(f"Total valid lines: {total_lines}")

# === 节点重编号并处理数据 ===
D = {}
ID = {}
max_index_before = -1
max_index_after = -1
num_lines_before = 0
num_lines_after = 0
buffer = []

print("Processing with progress bar...")

with open(txt_path, "r") as f:
    for line in tqdm(f, total=total_lines, unit="lines"):
        line = line.strip()
        if not line or line.startswith('%') or line.startswith('#'):
            continue

        tokens = line.split()
        # 跳过类似 "50912018 50912018 54054660" 的矩阵维度声明
        if is_mtx and len(tokens) == 3:
            continue
        if len(tokens) < 2:
            continue

        # 选择字段
        a, b = tokens[0], tokens[1]

        if not a.isdigit() or not b.isdigit():
            continue  # 忽略非数字行（如 matrix headers）

        if a == b:
            continue  # 自环跳过

        num_lines_before += 1

        if a not in D:
            ID[len(D)] = a
            D[a] = len(D)
        if b not in D:
            ID[len(D)] = b
            D[b] = len(D)

        buffer.append(f"{D[a]} {D[b]}\n")

        max_index_before = max(max_index_before, int(a), int(b))
        max_index_after = max(max_index_after, D[a], D[b])
        num_lines_after += 1

# === 写出处理结果 ===
print(f"Writing {len(buffer)} edges to {output_path} ...")
with open(output_path, "w") as f:
    f.writelines(buffer)

# === 输出信息 ===
print("\n=== Summary ===")
print("Original max index:", max_index_before)
print("Original edges:", num_lines_before)
print("Processed max index:", max_index_after)
print("Processed edges:", num_lines_after)
print("Output written to:", output_path)
