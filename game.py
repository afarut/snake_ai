import pygame
import numpy as np
from time import sleep
from main import *

MAP = Map2(MAP_COUNT_SIZE)

snake = Snake(MAP)


while True:
    MAP.draw()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                snake.change_direction(MOVE["LEFT"])
            elif event.key == pygame.K_RIGHT:
                snake.change_direction(MOVE["RIGHT"])
            elif event.key == pygame.K_UP:
                snake.change_direction(MOVE["UP"])
            elif event.key == pygame.K_DOWN:
                snake.change_direction(MOVE["DOWN"])

    reward, done = snake.move()
    text = font.render(f"Score: {snake.amount}", True, (255, 255, 255))  # Белый цвет
    screen.blit(text, (10, 10))  # Размещаем в левом верхнем углу
    pygame.display.flip()
    if reward == WALL_AWARD:
        raise Exception("Вы врезались")
    sleep(0.1)