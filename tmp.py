import random

input_file = 'C:\\data\\dataset\\hmdb51\\hmdb51_val_split_1_rawframes.txt'  # 输入文件路径
output_file = 'C:\\data\\dataset\\hmdb51\\hmdb51_target_split_1_rawframes.txt'  # 输出文件路径

samples = {}
with open(input_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # 跳过格式不正确的行

        # 假设格式为: str int int
        sample_str = parts[0]
        sample_int1 = int(parts[1])
        sample_int2 = int(parts[2])
        category = sample_int2  # 假设最后一个 int 是类别

        # 将样本存储在字典中，类别为键，样本为值
        if category not in samples:
            samples[category] = []
        samples[category].append(line.strip())

# 随机采样每种类别的一个样本
sampled_lines = []
for category, lines in samples.items():
    sampled_line = random.choice(lines)  # 随机选择一个样本
    sampled_lines.append(sampled_line)

# 写入新的数据文件
with open(output_file, 'w') as file:
    for line in sampled_lines:
        file.write(line + '\n')
