from os import path, makedirs, listdir
import shutil
import glob
import cv2


class DatasetPrep:
    @staticmethod
    def split_dataset(directory, test_proportion=20):
        '''
        Splits data set directory into test and train images accordingly to the given proportion ans an input
        parameter to the function.
        :param directory: root directory of an data set
        :param test_proportion: proportion of how data set should be split into test/train
        :return: None
        '''
        if not path.exists(directory):
            print("Path does not exist: ", directory)
            return

        total_file_count = listdir(directory)
        print("Total files: ", total_file_count)

        test_files_count = int(total_file_count * test_proportion / 100)
        train_files_count = total_file_count - test_files_count
        print("Test files counter: ", test_files_count)
        print("Train files count: ", train_files_count)

        makedirs("test")
        makedirs("train")
        counter = 0
        for filePath in glob.glob(directory + "\\*"):
            if counter < test_files_count:
                shutil.move(os.path.join(directory, filePath), os.path.join(directory, "/test"))
                print("Moving file {}, to directory: ".format(
                    os.path.join(directory, filePath), os.path.join(directory, "/test")))
            else:
                shutil.move(os.path.join(dir, filePath), os.path.join(directory, "/train"))
                print("Moving file {}, to directory: ".format(
                    os.path.join(directory, filePath), os.path.join(directory, "/train")))
        counter += 1

        print("All files moved successfully. Total files moved: ", counter)

    @staticmethod
    def transform_dataset(dataset_path):
        '''
        !!!Use with caution, as it overrides images!!!
        Transforms images in a given directory to black and white color images.
        :param dataset_path: path to the root of data set.
        :return: None
        '''
        for item in os.listdir(dataset_path):
            if os.path.isfile(dataset_path + "\\" + item):
                image = cv2.imread(dataset_path + "\\" + item)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.GaussianBlur(image, (5, 5), 0)

                status, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
                cv2.imwrite(dataset_path + "\\" + item, image)