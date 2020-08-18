import requests
import shutil
from os import path, makedirs, listdir

class datasetPrep:
    def __init__(self):
        pass

    def download_dataset(self, url, dir):
        if not path.exists(dir):
            makedirs(dir, exist_ok=True)

        response = requests.get(url, dir, stream=True)

        if response is not 200:
            print("Error: Unable to load data from url: ", url)

        print("Response from server: ", response)

        with open(dir, "wb") as file:
            for chunk in response.iter_content(chunk_size=256):
                file.write(chunk)

        print("Dataset downloaded with status success at directory: ", dir)

    def split_dataset(self, dir, test_proportion=20, override_dir=False):
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
        for file in os.listdir(self.dlPth):
            newfile = os.path.join(self.destPth, "name-of-new-file")
            shutil.move(os.path.join(self.dlPth, file), newfile)
