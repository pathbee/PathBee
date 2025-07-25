import os
import re

filename = "out.sx-mathoverflow"
dir1 = 'data/sx-mathoverflow'
dir2 = 'data/sx-mathoverflow-processed'
os.makedirs(dir2, exist_ok=True)

input_path = os.path.join(dir1, filename)
output_path = os.path.join(dir2, filename)

D = {}   # 原始ID → 新编号
ID = {}  # 新编号 → 原始ID

max_index_before = -1
max_index_after = -1
num_lines_before = 0
num_lines_after = 0

with open(input_path, "r") as f1, open(output_path, "w") as f2:
    for line in f1:
        line = line.strip()
        if not line or line.startswith('%'):
            continue

        tokens = re.split(r'\s+', line)  # ← 核心修改点，兼容空格和制表符
        if len(tokens) < 2:
            continue

        a, b = tokens[:2]
        num_lines_before += 1

        if a == b:
            continue

        if a not in D:
            D[a] = len(D)
            ID[D[a]] = a
        if b not in D:
            D[b] = len(D)
            ID[D[b]] = b

        a_idx, b_idx = D[a], D[b]
        f2.write(f"{a_idx} {b_idx}\n")

        try:
            max_index_before = max(max_index_before, int(a), int(b))
        except ValueError:
            pass

        max_index_after = max(max_index_after, a_idx, b_idx)
        num_lines_after += 1

print("✓ 原始边数:", num_lines_before)
print("✓ 处理后边数:", num_lines_after)
print("✓ 原始最大ID（如果是整数）:", max_index_before)
print("✓ 新编号最大索引:", max_index_after)
print("✓ 映射规模:", len(D), "个唯一节点")