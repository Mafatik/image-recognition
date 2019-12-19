import os
import copy 
import numpy as np
from PIL import Image


DATA_LEARN_PATH = './data/learn'
DATA_RECOGNIZE_PATH = './data/recognize'

INPUT_PICTURES_PATH = './input_pictures/'


class FileProcessor:
    def _parse_file_content(self, string):
        cols = string.find('\n')
        matrix = []
        string_copy = copy.copy(string)
        string_copy = string.replace('\n', '')
        for row_index in range(int(len(string) / cols)):
            row = string_copy[row_index * cols: row_index * cols + cols].split(' ')
            row = [int(elem) for elem in row]
            matrix.append(row)
        
        return matrix

    def _to_picture(self, matrix, path_to_save, name):
        matrix_copy = copy.copy(matrix)
        matrix_copy = np.asarray(matrix_copy).astype('uint8')
        matrix_copy[matrix_copy == 0] = 255

        image = Image.fromarray(matrix_copy)
        image.save(os.path.join(path_to_save, name + '.jpg'))

    def _prapare_folder(self, path):
        folder_name = os.path.basename(os.path.normpath(path))
        dir_name = os.path.join(INPUT_PICTURES_PATH, folder_name)
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        file_names = [name for name in os.listdir(path)]

        images = {}
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            file = open(file_path, 'r')
            file_content = file.read()
            matrix = self._parse_file_content(self, file_content)

            self._to_picture(self, matrix, dir_name, file_name)

            matrix = np.array(matrix)
            matrix[matrix == 0] = -1
            matrix = matrix.flatten()

            images[file_name] = matrix

        return images

    @classmethod
    def prepare(self):
        learn = self._prapare_folder(self, DATA_LEARN_PATH)
        recognize = self._prapare_folder(self, DATA_RECOGNIZE_PATH)

        return (learn, recognize)
