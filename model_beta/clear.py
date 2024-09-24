import os
import shutil

input_dir = r'model_beta\model_beta\input_folder'
output_dir = r'model_beta\model_beta\output_folder'
detected_frame_dir = r'model_beta\model_beta\detected_frame_folder'

# Clear input folder
for file in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# Clear output folder
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# Clear detected frame folder subfolders
for root, dirs, files in os.walk(detected_frame_dir):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        shutil.rmtree(dir_path)
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)

print('Cleared all files and folders in input_folder, output_folder, and detected_frame_folder')