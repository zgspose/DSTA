import os
import glob

def rename(folder_path):
    # Get the paths of all files in the folder
    files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))  # You can modify the file type as needed (e.g., `*.jpg`)
    for index, file_path in enumerate(files, start=1):
        # Get the file extension.
        file_extension = os.path.splitext(file_path)[1]
        # Convert to posetrack18 data format
        new_name = f"{folder_path}/{index:08d}{file_extension}"
        # Rename
        os.rename(file_path, new_name)
    print('rename end')

# test
if __name__ == '__main__':
    folder_path = './path'
    rename(folder_path)
