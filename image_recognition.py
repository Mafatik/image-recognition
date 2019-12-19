from colorama import init
from file_processor import FileProcessor
from neural_network import NeuralNetwork

if __name__ == '__main__':
    init()

    learning_images, recognition_images = FileProcessor.prepare()
    neural_network = NeuralNetwork(learning_images, recognition_images)

    neural_network.learn()
    neural_network.recognize()

    k=input("press close to exit") 
