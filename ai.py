from main import *
from collections import deque
import random
import torch
import torch.nn as nn
import pygame


torch.manual_seed(42)

def preprocces(x):
    x = x.clone()
    x[x == WALL_AWARD] = 50
    x[x == FRUIT_AWARD] = 100
    x[x == -2] = 0
    x[x == -3] = 255
    x /= 255
    return x.unsqueeze(0).unsqueeze(0)


class BufferRelay:
    def __init__(self, batch_size=32, maxlen=10000):
        self.deque = deque([], maxlen=maxlen)
        self.maxlen = maxlen
        self.batch_size = batch_size

    def push(self, state, action, next_state, reward):
        self.deque.append((state, action, next_state, reward))
        for _ in range(3):
            tmp_state, tmp_action, tmp_next_state, tmp_reward = self.deque[-1]
            self.deque.append((torch.rot90(tmp_state, k=1, dims=(-1, -2)), 
                               (tmp_action + 1) % 4, 
                               torch.rot90(tmp_next_state, k=1, dims=(-1, -2)) if tmp_next_state is not None else None, 
                               reward))
        state_1 = torch.flip(state, dims=[-1])
        if next_state is None:
            next_state_1 = None
        else:
            next_state_1 = torch.flip(next_state, dims=[-1])
        if action == 1:
            action_1 = 3
        elif action == 3:
            action_1 = 1
        else:
            action_1 = action
        self.deque.append((state_1, action_1, next_state_1, reward))
        for _ in range(3):
            tmp_state, tmp_action, tmp_next_state, tmp_reward = self.deque[-1]
            self.deque.append((torch.rot90(tmp_state, k=1, dims=(-1, -2)), 
                               (tmp_action + 1) % 4, 
                               torch.rot90(tmp_next_state, k=1, dims=(-1, -2)) if tmp_next_state is not None else None, 
                               reward))
    
    def sample(self):
        batch = random.sample(self.deque, self.batch_size)
        non_final_mask = []
        for i in range(self.batch_size):
            if batch[i][2] is not None:
                non_final_mask.append(i)
        states, actions, next_states_tmp, rewards = list(zip(*batch))
        next_states = []
        for i in next_states_tmp:
            if i is not None:
                next_states.append(i)
        states = torch.cat(states, dim=0)
        actions = torch.LongTensor(actions)
        next_states = torch.cat(next_states, dim=0)
        rewards = torch.FloatTensor(rewards)
        return states, actions, next_states, rewards, non_final_mask
    
    def __len__(self):
        return len(self.deque)


class SnakeCNN(nn.Module):
    def __init__(self):
        super(SnakeCNN, self).__init__()
        self.conv =  nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(6, 6, kernel_size=5, padding=1),
            nn.MaxPool2d(2),
            
            # nn.Conv2d(16, 16, kernel_size=3),
        )
        self.fc = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

BATCH_SIZE = 64
buffer = BufferRelay(batch_size=BATCH_SIZE)

target_net = SnakeCNN()
policy_net = SnakeCNN()

policy_net.load_state_dict(torch.load("model_scripted.pt", weights_only=True))
target_net.load_state_dict(policy_net.state_dict())

MAP = Map2(MAP_COUNT_SIZE)
snake = Snake(MAP)

moves = (MOVE["UP"], MOVE["RIGHT"], MOVE["DOWN"], MOVE["LEFT"])

epoch = 0
step = 0
eps = 0.1
GAMMA = 0.9
TAU = 0.005

next_state = preprocces(MAP.map)
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=2e-3, weight_decay=0.001)
life_times = [0]
amount = [0]
try:
    while True:
        step += 1
        state = next_state
        pred = policy_net(state)
        if random.random() > eps:
            action = pred.argmax().item()
        else:
            action = random.randint(0, 3)
        snake.change_direction(moves[action])
        reward, done = snake.move()
        if done:
            next_state = None
        else:
            next_state = preprocces(MAP.map)

        buffer.push(state=state, action=action, next_state=next_state, reward=reward)

        if len(buffer) > BATCH_SIZE and False:
            states, actions, next_states, rewards, non_final_mask = buffer.sample()
            target = torch.zeros(BATCH_SIZE)
            with torch.no_grad():
                target[non_final_mask] = GAMMA * target_net(next_states).max(1).values
            target += rewards

            pred = torch.gather(policy_net(states), 1, actions.unsqueeze(1)).squeeze()

            loss = criterion(pred, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

            target_net.load_state_dict(target_net_state_dict)




        if next_state is None:
            eps = max(0, 0 * round(eps * 0.995, 4))
            life_times.append(snake.life_time)
            amount.append(snake.amount)
            MAP = Map2(MAP_COUNT_SIZE)
            snake = Snake(MAP)
            next_state = preprocces(MAP.map)
        
        if step % 1000 == 0:
            print(f"Epoch: {step}\tAmount:{(sum(amount) / len(amount)):.4f}\tLife Time: {(sum(life_times) / len(life_times)):.4f}\tEpsilon: {eps}")
            # amount = [0]
            # life_times = [0]
        if step % 100000 == 0:
            torch.save(policy_net.state_dict(), f"model_scripted_{step}.pt")

        if GAME_MODE:
            MAP.draw()
            text = font.render(f"Score: {snake.amount}", True, (255, 255, 255))  # Белый цвет
            screen.blit(text, (10, 10))  # Размещаем в левом верхнем углу
            pygame.display.flip()

except KeyboardInterrupt:
    print("Завершено")

torch.save(policy_net.state_dict(), "model_scripted.pt")



import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


tensor_gray = MAP.map  # Одноканальное изображение

tensor_gray[tensor_gray == WALL_AWARD] = 50
tensor_gray[tensor_gray == FRUIT_AWARD] = 100
tensor_gray[tensor_gray == -2] = 180
tensor_gray[tensor_gray == -3] = 255
tensor_gray /= 255
from PIL import Image
import torchvision.transforms as transforms


plt.imshow(tensor_gray.squeeze())
plt.savefig(f"./ai_image/map.png")




def get_conv_activations(model, x):
    activations = []
    
    for layer in model.conv:
        x = layer(x)  # Пропускаем вход через слой
        if isinstance(layer, nn.Softmax2d) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LeakyReLU) or  isinstance(layer, nn.MaxPool2d):  # Если слой - сверточный, сохраняем выход
            activations.append(x.detach())

    return activations


def plot_activations(activations, layer_names):
    for i, (activation, name) in enumerate(zip(activations, layer_names)):
        num_filters = activation.shape[1]  # Количество фильтров (карт активаций)
        fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
        fig.suptitle(f"Активации после {name}")
        
        if num_filters == 1:
            axes = [axes]
        
        for j in range(num_filters):  # Перебираем карты активаций
            ax = axes[j]
            im = ax.imshow(activation[0, j].cpu().numpy(), cmap="viridis")
            ax.axis("off")
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.savefig(f"./ai_image/activation_layer_{i}.png", bbox_inches='tight', dpi=300)
        plt.close()  # Закрываем фигуру
 
 
activations = get_conv_activations(policy_net, tensor_gray.resize(1, 1, 18, 18))

# Визуализируем активации
plot_activations(activations, ["Conv1", "MaxPool2d", "Conv2", "MaxPool2d2"])