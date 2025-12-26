import os
import glob

folder_path = "output_ppo/state"
pattern = os.path.join(folder_path, "**", "*-state.csv")

changed, skipped = 0, 0

for file_path in glob.iglob(pattern, recursive=True):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        lines = f.readlines()
    if not lines:
        skipped += 1
        continue

    # 只处理第一行表头：去掉BOM/空白后精确替换
    header = lines[0].rstrip("\r\n")
    cols = [c.strip().lstrip("\ufeff") for c in header.split(",")]
    new_cols = ["agent_jeck" if c == "agent_jerk" else c for c in cols]
    new_header = ",".join(new_cols)

    if new_header != header:
        lines[0] = new_header + "\n"
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        changed += 1
        print(f"已修改: {file_path}")
    else:
        skipped += 1
        print(f"无需修改: {file_path}")

print(f"完成。修改 {changed} 个文件，跳过 {skipped} 个。")
