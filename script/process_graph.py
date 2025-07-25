import os

filename = "out.sx-mathoverflow"
dir1 = 'data/sx-mathoverflow'
dir2 = 'data/sx-mathoverflow-processed'
os.makedirs(dir2, exist_ok=True)

input_path = os.path.join(dir1, filename)
output_path = os.path.join(dir2, filename)

D = {}
ID = {}

max_index_before = -1
max_index_after = -1
num_lines_before = 0
num_lines_after = 0

with open(input_path, "r") as f1, open(output_path, "w") as f2:
    for line in f1:
        line = line.strip()
        if not line or line.startswith('%'):
            continue  # 跳过注释或空行

        tokens = line.split()
        if len(tokens) != 2:
            continue  # 跳过不合法的行

        a, b = tokens
        num_lines_before += 1

        if a == b:
            continue  # 自环跳过

        if a not in D:
            ID[len(D)] = a
            D[a] = len(D)
        if b not in D:
            ID[len(D)] = b
            D[b] = len(D)

        f2.write(f"{D[a]} {D[b]}\n")

        max_index_before = max(max_index_before, int(a), int(b))
        max_index_after = max(max_index_after, D[a], D[b])
        num_lines_after += 1

print("处理前的最大索引值:", max_index_before, "处理前的行数:", num_lines_before)
print("处理后的最大索引值:", max_index_after, "处理后的行数:", num_lines_after)
