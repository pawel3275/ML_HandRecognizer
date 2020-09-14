from os import path, makedirs, listdir
import requests
import shutil
import glob


class DatasetPrep:
    def __init__(self):
        self.current_directory = os.getcwd()
        pass

    @staticmethod
    def download_dataset(url, directory):
        if not path.exists(directory):
            makedirs(directory, exist_ok=True)

        response = requests.get(url, directory, stream=True)
        print(type(response))
        if response is not 200:
            print("Error: Unable to load data from url: ", url)

        print("Response from server: ", response)

        with open(directory, "wb") as file:
            for chunk in response.iter_content(chunk_size=256):
                file.write(chunk)

        print("Dataset downloaded with status success at directory: ", directory)

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
        for filePath in glob.glob(dir + "\\*"):
            if counter < test_files_count:
                shutil.move(os.path.join(dir, filePath), os.path.join(dir, "/test"))
                print("Moving file {}, to directory: ".format(os.path.join(dir, filePath), os.path.join(dir, "/test")))
            else:
                shutil.move(os.path.join(dir, filePath), os.path.join(dir, "/train"))
                print("Moving file {}, to directory: ".format(os.path.join(dir, filePath), os.path.join(dir, "/train")))
        counter += 1

        print("All files moved successfully. Total files moved: ", counter)

    @staticmethod
    def transform_dataset(dataset_path):
        for item in os.listdir(dataset_path):
            if os.path.isfile(dataset_path + "\\" + item):
                image = cv2.imread(dataset_path + "\\" + item)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.GaussianBlur(image, (5, 5), 0)

                status, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
                cv2.imwrite(dataset_path + "\\" + item, image)