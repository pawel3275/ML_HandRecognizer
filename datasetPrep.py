import requests
import shutil
import glob
from os import path, makedirs, listdir


class DatasetPrep:
    def __init__(self):
        self.current_directory = os.getcwd()
        pass

    @staticmethod
    def download_dataset(url, dir):
        if not path.exists(dir):
            makedirs(dir, exist_ok=True)

        response = requests.get(url, dir, stream=True)
        print(type(response))
        if response is not 200:
            print("Error: Unable to load data from url: ", url)

        print("Response from server: ", response)

        with open(dir, "wb") as file:
            for chunk in response.iter_content(chunk_size=256):
                file.write(chunk)

        print("Dataset downloaded with status success at directory: ", dir)

    @staticmethod
    def split_dataset(dir, test_proportion=20, override_dir=False):
        if not path.exists(dir):
            print("Path does not exist: ", dir)
            return

        total_file_count = listdir(dir)
        print("Total files: ", total_file_count)

        test_files_count = int(total_file_count * test_proportion / 100)
        train_files_count = total_file_count - test_files_count
        print("Test files counter: ", test_files_count)
        print("Train files count: ", train_files_count)

        makedirs("test")
        makedirs("train")
        counter = 0
        for filePath in glob.glob(dir + '\\*'):
            if counter < test_files_count:
                shutil.move(os.path.join(dir, filePath), os.path.join(dir, "/test"))
                print("Moving file {}, to directory: ".format(os.path.join(dir, filePath), os.path.join(dir, "/test")))
            else:
                shutil.move(os.path.join(dir, filePath), os.path.join(dir, "/train"))
                print("Moving file {}, to directory: ".format(os.path.join(dir, filePath), os.path.join(dir, "/train")))
        counter += 1

        print("All files moved successfully. Total files moved: ", counter)

