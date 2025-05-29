import os
import random
import string

def generate_random_base64_filename(length=20):
    # 生成一个包含字母和数字的随机字符串
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return random_string

def rename_files_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 确保是文件而不是子文件夹
        if os.path.isfile(file_path):
            # 获取文件扩展名
            file_name, file_extension = os.path.splitext(filename)
            # 生成新的Base64文件名（仅包含字母和数字）
            new_file_name = generate_random_base64_filename() + file_extension
            new_file_path = os.path.join(folder_path, new_file_name)
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"文件 {filename} 已重命名为 {new_file_name}")

# 示例：修改文件夹内所有文件名
folder_path = "./Tina"  # 这里填写目标文件夹路径
rename_files_in_folder(folder_path)
