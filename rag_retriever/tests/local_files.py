import os


# 获取本地文件目录文件列表
def get_test_files():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    local_files_path = os.path.join(current_file_dir, "files")
    entries = os.listdir(local_files_path)
    absolute_paths = []

    for entry in entries:
        file_path = os.path.join(local_files_path, entry)
        if os.path.isfile(file_path):
            absolute_path = os.path.abspath(file_path)
            absolute_paths.append(absolute_path)

    return absolute_paths
