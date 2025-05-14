import os
import shutil

def flatten_and_copy(directory, target_directory):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 获取所有文件的路径
    for root, dirs, files in os.walk(directory):
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_directory, file)

            # 如果目标文件夹已有同名文件，可以选择覆盖或修改文件名
            if os.path.exists(target_file):
                base, extension = os.path.splitext(file)
                target_file = os.path.join(target_directory, base + "_1" + extension)

            # 复制文件到目标文件夹
            shutil.copy(source_file, target_file)
            print(f"文件 {source_file} 已复制到 {target_file}")

# 使用例子
source_directory = "./pics/Pics"
target_directory = "./pics/imgs"
flatten_and_copy(source_directory, target_directory)
