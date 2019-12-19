import numpy as np
from colorama import Fore

class NeuralNetwork:
    def __init__(self, learning_images, recognition_images):
        self.learning_images = learning_images
        self.recognition_images = recognition_images
        self.image_len = len(learning_images[list(learning_images)[0]])
        self.w = np.zeros((self.image_len, self.image_len))

    def learn(self):
        for name in self.learning_images.keys():
            transposed = np.expand_dims(np.transpose(self.learning_images[name]), axis=1)
            vector =  np.expand_dims(self.learning_images[name], axis = 0)
            new_w = np.matmul(
                        transposed,
                        vector
            )
            self.w = np.add(self.w, new_w)

        np.fill_diagonal(self.w, [0. for x in range(self.image_len)])

    def activate_sign(self, vector):
        vector[vector >= 0.] = 1.
        vector[vector < 0.] = -1.

        return vector

    def recognize(self):
        for r_name in self.recognition_images.keys():
            print(Fore.BLUE, '*************', r_name, end='')

            prev_vector = []
            out = np.expand_dims(np.transpose(self.recognition_images[r_name]), axis=1)

            while not np.array_equal(prev_vector, out):
                summa = np.matmul(self.w, out)
                out = self.activate_sign(summa)
                prev_vector = out

            out = np.transpose(out)
            out = out[0]
            euqals = 'N/A'
            for l_name in self.learning_images.keys():
                if np.array_equal(self.learning_images[l_name], out):
                    equals = l_name
                    break

            print(Fore.BLUE, ' equals: ', l_name)
