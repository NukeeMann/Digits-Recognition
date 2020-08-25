from Neural_Network import *
import pygame
import random


# Main window object
class Window:
    screen = None
    nn = None
    rect = None
    results = None

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 585))
        self.screen.fill((0, 0, 0))
        pygame.display.set_caption("Handwriting recognition")

        # Drawing chequerboard for drawing
        for y in range(0, 28):
            for x in range(0, 28):
                pygame.draw.rect(self.screen, (255, 255, 255), (17 * y + 17, 17 * x + 17, 16, 16))

        pygame.draw.rect(self.screen, (255, 255, 255), (17, 510, 150, 60))  # On click 217, 217, 217
        pygame.draw.rect(self.screen, (255, 255, 255), (180, 510, 150, 60))
        pygame.draw.rect(self.screen, (153, 0, 153), (343, 510, 150, 60))  # On click 102, 0, 102

        font = pygame.font.SysFont('Corbel', 35)

        text_clear = font.render('Clear', True, (0, 0, 0))
        text_train = font.render('Train', True, (0, 0, 0))
        text_predict = font.render('Predict', True, (0, 0, 0))
        self.screen.blit(text_clear, (55, 525))
        self.screen.blit(text_train, (218, 525))
        self.screen.blit(text_predict, (370, 525))

        font_2 = pygame.font.SysFont('Corbel', 40, bold=True)
        text_results = font_2.render('Results:', True, (255, 255, 255))
        self.screen.blit(text_results, (580, 17))

        self.rect = np.zeros(shape=(1, 28, 28), dtype=float)
        self.results = np.zeros(shape=(1, 10), dtype=float)

        self.update_results()

    def color_in(self, x, y, rect):
        # Coloring pixel under cursor
        rect[0][y][x] = random.uniform(0.9, 1.0)
        pygame.draw.rect(self.screen, (0, 0, 0), (17 * x + 17, 17 * y + 17, 16, 16))

        # Coloring pixels around our mouse position
        for y_grey in range(-1, 2):
            for x_grey in range(-1, 2):
                if 0 <= y - y_grey < 28 and 0 <= x - x_grey < 28 and (x_grey != 0 or y_grey != 0):
                    if rect[0][y - y_grey][x - x_grey] < 0.9:
                        grey_scale = 50

                        if x_grey == 0 or y_grey == 0:
                            rect[0][y - y_grey][x - x_grey] = random.uniform(0.7, 0.89)
                            pygame.draw.rect(self.screen,
                                             (grey_scale, grey_scale, grey_scale),
                                             (17 * (x - x_grey) + 17, 17 * (y - y_grey) + 17, 16, 16))
                        if x_grey != 0 and y_grey != 0 and rect[0][y - y_grey][x - x_grey] < 0.3:
                            rect[0][y - y_grey][x - x_grey] = random.uniform(0.4, 0.6)
                            pygame.draw.rect(self.screen,
                                             (grey_scale + 40, grey_scale + 40, grey_scale + 40),
                                             (17 * (x - x_grey) + 17, 17 * (y - y_grey) + 17, 16, 16))

        return rect

    def update_results(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (580, 80, 150, 330))
        font = pygame.font.SysFont('Arial Black', 25)
        for iterator in range(0, 10):
            compact_text = str(iterator) + " - " + str(round(self.results[0][iterator], 3))
            if self.results[0][iterator] == max(self.results[0]) != 0.0:
                text = font.render(compact_text, True, (0, 254, 0))
            else:
                text = font.render(compact_text, True, (254, 254, 254))
            self.screen.blit(text, (600, 80 + iterator * 32))
        return

    def button_clear(self):
        for y in range(0, 28):
            for x in range(0, 28):
                pygame.draw.rect(self.screen, (255, 255, 255), (17 * y + 17, 17 * x + 17, 16, 16))

        self.rect = np.zeros(shape=(1, 28, 28), dtype=float)

    def button_train(self, n_nodes, n_epos):
        # Window to set number of nodes and eposes
        self.set_neural_network(num_od_nodes=n_nodes, epos=n_epos)

    def button_predict(self):
        self.predict()
        # Update results
        self.update_results()

    def set_neural_network(self, num_od_nodes=16, epos=5):
        # Declaring our neural network
        self.nn = NeuralNetwork(num_od_nodes)

        # Loading database
        X_train, Y_train, X_test, Y_test = self.nn.load_database()

        # Calling function to train our neural network in given epos
        end, start = self.nn.nn_training(X_train, Y_train, X_test, Y_test, epos)

        # Print accuracy
        print("Handwriting recognition accuracy")
        print(str(self.nn.acc_fn(X_test, Y_test) * 100) + "%")
        print("Time: " + str(end - start))

    def predict(self):
        if self.nn is None:
            return
        self.results = self.nn.pr_fn([self.rect])
