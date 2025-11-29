import pandas as pd

# 替换为你的文件路径
file_path = r"F:\MUSA5500\data\analysis_data\all_points_full.csv"

# 读取 CSV
df = pd.read_csv(file_path)

# 去重
unique_rows = df.drop_duplicates()

# 打印不重复的行数
print("不重复行数:", len(unique_rows))
