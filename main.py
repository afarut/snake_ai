import pygame
import torch
from time import sleep
import numpy as np
import random



MAP_COUNT_SIZE = 18
NOTHING_COLOR = (125, 125, 125)
BLOCK_SIZE = 40
MARGIN_SIZE = 2
RED = (125, 50, 50)
GREEN = (50, 125, 50)
BLACK = (0, 0, 0)
SNAKE_COLOR = (50, 50, 150)
SNAKE_HEAD_COLOR = (50, 50, 250)
MOVE = {
    "UP": torch.tensor([-1, 0]),
    "DOWN": torch.tensor([1, 0]),
    "RIGHT": torch.tensor([0, 1]),
    "LEFT": torch.tensor([0, -1])
}


moves = (MOVE["UP"], MOVE["RIGHT"], MOVE["DOWN"], MOVE["LEFT"])

FRUIT_AWARD = 1
WALL_AWARD = -5
SNAKE_HEAD = -3
SNAKE_BODY = -2
GAME_MODE = 1

size = [MARGIN_SIZE * (MAP_COUNT_SIZE + 1) + BLOCK_SIZE * MAP_COUNT_SIZE,
        MARGIN_SIZE * (MAP_COUNT_SIZE + 1) + BLOCK_SIZE * MAP_COUNT_SIZE]

if GAME_MODE:
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Змейка")
    pygame.font.init()
    font = pygame.font.Font(None, 36)


class Map:
    def __init__(self, MAP_COUNT_SIZE, GAME_MODE=GAME_MODE):
        self.size = MAP_COUNT_SIZE
        self.margin = 3
        self.GAME_MODE = GAME_MODE
        self.map = torch.zeros((MAP_COUNT_SIZE, MAP_COUNT_SIZE), dtype=torch.float32)
        self.map[17, 17] = FRUIT_AWARD
        self.map[9, 9] = FRUIT_AWARD

    def colors(self, value):
        v = int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        if v == SNAKE_BODY:
            return SNAKE_COLOR
        elif v == FRUIT_AWARD:
            return GREEN
        elif v == WALL_AWARD:
            return BLACK
        elif v == SNAKE_HEAD:
            return SNAKE_HEAD_COLOR
        return NOTHING_COLOR

    def generate(self):
        # Генерируем одномерный тензор длины self.size с заданными значениями и вероятностями
        values = torch.tensor([WALL_AWARD, FRUIT_AWARD, 0], dtype=torch.int)
        probabilities = torch.tensor([0.05, 0.01, 0.94], dtype=torch.float)
        indices = torch.multinomial(probabilities, self.size, replacement=True)
        result = values[indices]
        return result

    def draw(self):
        if not self.GAME_MODE:
            return
        for i in range(self.size):
            for j in range(self.size):
                pygame.draw.rect(
                    screen,
                    self.colors(self.map[i, j]),
                    [j * BLOCK_SIZE + (j + 1) * MARGIN_SIZE,
                     i * BLOCK_SIZE + (i + 1) * MARGIN_SIZE,
                     BLOCK_SIZE, BLOCK_SIZE]
                )

    def snake_draw(self, coords):
        award = 0
        if self.size - coords[-1][1] < self.margin + 1:
            # Сдвигаем все координаты по столбцам
            for i in range(len(coords)):
                coords[i][1] -= 1
            new_data = self.generate().reshape(self.size, 1)
            tmp = self.map[:,0]
            award = 0.005 * torch.sum(tmp == 0).item()
            self.map = self.map[:, 1:]
            self.map = torch.cat((self.map, new_data), dim=1)
        elif self.size - coords[-1][0] < self.margin + 1:
            for i in range(len(coords)):
                coords[i][0] -= 1
            new_data = self.generate().reshape(1, self.size)
            tmp = self.map[0,:]
            award = 0.005 * torch.sum(tmp == 0).item()
            self.map = self.map[1:, :]
            self.map = torch.cat((self.map, new_data), dim=0)
        elif coords[-1][0] < self.margin:
            for i in range(len(coords)):
                coords[i][0] += 1
            new_data = self.generate().reshape(1, self.size)
            tmp = self.map[-1,:]
            award = 0.005 * torch.sum(tmp == 0).item()
            self.map = self.map[:-1, :]
            self.map = torch.cat((new_data, self.map), dim=0)
        elif coords[-1][1] < self.margin:
            for i in range(len(coords)):
                coords[i][1] += 1
            new_data = self.generate().reshape(self.size, 1)
            tmp = self.map[:,-1]
            award = 0.005 * torch.sum(tmp == 0).item()
            self.map = self.map[:, :-1]
            self.map = torch.cat((new_data, self.map), dim=1)
        else:
            pass

        for coord in coords:
            if self.map[coord[0], coord[1]] != WALL_AWARD:
                self.map[coord[0], coord[1]] = SNAKE_BODY
        if self.map[coords[-1][0], coords[-1][1]] != WALL_AWARD:
            self.map[coords[-1][0], coords[-1][1]] = SNAKE_HEAD
        self.draw()
        return award

    def fillna(self):
        self.map[self.map == SNAKE_BODY] = 0
        return self


class Map2(Map):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map[0, :] = WALL_AWARD
        self.map[:, 0] = WALL_AWARD
        self.map[-1, :] = WALL_AWARD
        self.map[:, -1] = WALL_AWARD

        row = np.random.randint(1, self.map.shape[0]-1)
        col = np.random.randint(1, self.map.shape[1]-1)
        while self.map[row, col] != 0:
            row = np.random.randint(1, self.map.shape[0]-1)
            col = np.random.randint(1, self.map.shape[1]-1)
        self.map[row, col] = FRUIT_AWARD
        # self.map[12, 12] = WALL_AWARD
        # self.map[13, 13] = WALL_AWARD
        # self.map[12, 14] = WALL_AWARD
        # self.map[12, 5] = WALL_AWARD
        # self.map[13, 6] = WALL_AWARD
        # self.map[14, 7] = WALL_AWARD
    
    def snake_draw(self, coords):
        if self.map[coords[-1][0], coords[-1][1]] == FRUIT_AWARD:
            self.map[coords[-1][0], coords[-1][1]] = SNAKE_HEAD

            row = np.random.randint(1, self.map.shape[0]-1)
            col = np.random.randint(1, self.map.shape[1]-1)

            while self.map[row, col] != 0:
                row = np.random.randint(1, self.map.shape[0]-1)
                col = np.random.randint(1, self.map.shape[1]-1)

            self.map[row, col] = FRUIT_AWARD

        for coord in coords:
            if self.map[coord[0], coord[1]] != WALL_AWARD:
                self.map[coord[0], coord[1]] = SNAKE_BODY
        if self.map[coords[-1][0], coords[-1][1]] != WALL_AWARD:
            self.map[coords[-1][0], coords[-1][1]] = SNAKE_HEAD
        return 0



class Snake:
    def __init__(self, map_obj):
        self.move_dir = MOVE["RIGHT"]
        self.map = map_obj
        start_x = random.randint(3, 12)
        start_y = random.randint(3, 12)
        coord = [[start_x, start_y - i] for i in range(4)][::-1]
        self.coord = coord

        self.amount = 0
        self.life_time = 0
        self.loader = 0

    def move(self):
        self.life_time += 1
        self.loader += 1
        done = False
        self.map.fillna()
        next_step = self.coord[-1].copy()
        next_step[0] += int(self.move_dir[0].item())
        next_step[1] += int(self.move_dir[1].item())
        self.coord.append(next_step)
        self.coord.pop(0)
        award = self.map.map[next_step[0], next_step[1]].item()
        if award == WALL_AWARD:
            done = True
            self.amount += 0
        elif award == FRUIT_AWARD:
            self.amount += award
            self.loader = 0
        done = done or (self.loader > 50)
        tmp_award = self.map.snake_draw(self.coord)
        if award == WALL_AWARD or award == FRUIT_AWARD:
            return award + tmp_award, done
        return tmp_award, done

    def change_direction(self, dir):
        self.move_dir = dir
