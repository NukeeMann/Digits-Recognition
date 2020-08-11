from timeit import default_timer as timer
from Interface import *

if __name__ == "__main__":
    number_of_nodes = 80
    training_epos = 20
    program = Window()

    predict_b = False
    start_time = 0.0
    end_time = 0.0
    # Main loop for drawing
    loop = True
    while loop:
        try:
            if predict_b is True:
                end_time = timer()
                if end_time - start_time > 0.3:
                    predict_b = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    loop = False

            px, py = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed() == (1, 0, 0):
                # Checking for buttons
                if 510 < py < 570:
                    if 17 < px < 167:
                        program.button_clear()
                    elif 180 < px < 330:
                        program.button_train(number_of_nodes, training_epos)
                    elif 343 < px < 493 and predict_b is False:
                        start_time = timer()
                        predict_b = True
                        program.button_predict()

                # Checking for colored rectangles
                if px - 17 > 0 and py - 17 > 0:
                    px = round((px - 17) / 17)
                    py = round((py - 17) / 17)
                    if 0 <= px < 28 and 0 <= py < 28:
                        program.rect = program.color_in(px, py, program.rect)

            pygame.display.update()
        except Exception as e:
            print(e)
    pygame.quit()
