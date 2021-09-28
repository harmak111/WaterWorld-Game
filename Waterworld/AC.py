import numpy as np
import matplotlib.pyplot as plt
import game
import itertools
import random
from game import *
import pygame
from tqdm import tqdm

waterworld = game.Player(width=50, height=50, velocity=10)
index = np.zeros(1000)
rewards = np.zeros(1000)


def softmax_stable(posx, posy, c, theta):
    c_state = [(posx / (50 + 50)), (posy / (50 + 50))]
    b = np.cos(np.pi * np.dot(c_state, c.transpose()))
    act0 = np.append(b, np.zeros(48))
    act1 = np.append((np.append(np.zeros(16), b)), np.zeros(32))
    act2 = np.append((np.append(np.zeros(32), b)), np.zeros(16))
    act3 = np.append(np.zeros(48), b)
    act = np.array([act0, act1, act2, act3])
    x = np.dot(theta, act.transpose())
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


def softmax(posx, posy, c, theta):
    c_state = [(posx / (50 + 50)), (posy / (50 + 50))]
    b = np.cos(np.pi * np.dot(c_state, c.transpose()))
    act0 = np.append(b, np.zeros(48))
    act1 = np.append((np.append(np.zeros(16), b)), np.zeros(32))
    act2 = np.append((np.append(np.zeros(32), b)), np.zeros(16))
    act3 = np.append(np.zeros(48), b)
    act = np.array([act0, act1, act2, act3])
    x = np.dot(theta, act.transpose())
    exp = np.exp(x)
    return exp / np.sum(exp)


def fourier_basis(posx, posy, c):
    c_state = [(posx / (50 + 50)), (posy / (50 + 50))]
    return np.cos(np.pi * np.dot(c_state, c.transpose()))


def ac(episodes):
    index = np.zeros(episodes)
    lambda_theta = 0.7
    lambda_weight = 0.5
    alpha_theta = 0.0001
    alpha_weight = 0.0001
    gamma = 1.0
    theta = np.zeros(64)
    weight = np.zeros(16)
    rewards = np.zeros(episodes)
    ls = list(itertools.product([0, 1, 2, 3], repeat=2))
    constant = np.asarray(ls)
    for epi in tqdm(range(episodes)):
        print(epi)
        Player.goal_reached = False
        I = 1.0
        gameover = True
        pos_x = 0
        pos_y = 0
        enemy_x = 60
        enemy_y = 50
        z_theta = np.zeros(64)
        z_weight = np.zeros(16)
        ind = 0
        # print("first:"+str(pos_x))
        states = []
        states_enemy = []
        while gameover:
            position = np.array((pos_x, pos_y))
            enemy_position = np.array((enemy_x, enemy_y))

            ind += 1
            sm = softmax_stable(pos_x, pos_y, constant, theta)  # action probability
            action = np.random.choice(waterworld.actions, p=sm)  # taking an action

            # Next state and reward
            next_pos_x, next_pos_y, goal_reached, next_e_pos_x, next_e_pos_y, next_reward = waterworld.movement(pos_x,
                                                                                                                pos_y,
                                                                                                                action,
                                                                                                                enemy_x,
                                                                                                                enemy_y)
            states.append(position)  # adding position to a list
            index[epi] = ind

            rewards[epi] += next_reward
            states_enemy.append(enemy_position)  # adding enemy position to a list
            c = fourier_basis(pos_x, pos_y, constant)
            c_next = fourier_basis(next_pos_x, next_pos_y, constant)
            value_current = np.dot(weight, c)
            value_next = np.dot(weight, c_next)

            if action == "up":
                a = 0
            if action == "down":
                a = 1
            if action == "left":
                a = 2
            if action == "right":
                a = 3

            act0 = np.append(c, np.zeros(48))
            act1 = np.append((np.append(np.zeros(16), c)), np.zeros(32))
            act2 = np.append((np.append(np.zeros(32), c)), np.zeros(16))
            act3 = np.append(np.zeros(48), c)
            act = np.array([act0, act1, act2, act3])

            # print(act[a]+1)
            if goal_reached:
                delta = next_reward - value_current  # if S' is terminal
                gameover = False
            else:
                delta = next_reward + gamma * value_next - value_current  # if still running

            z_weight = gamma * lambda_weight * z_weight + c  # d-component eligibility trace vector

            z_theta = gamma * lambda_theta * z_theta + I * (
                    (act[a] + 1) - np.dot(sm, act))  # d'-component eligibility trace vector

            weight += alpha_weight * delta * z_weight  # Updates weight
            theta += alpha_theta * delta * z_theta  # Updates theta

            I *= gamma

            pos_x = next_pos_x  # Assign new values to x and y values of player and enemy
            pos_y = next_pos_y
            enemy_x = next_e_pos_x
            enemy_y = next_e_pos_y

    return index, rewards, states, states_enemy


episodes = 1000
timesteps = np.zeros(episodes)
reward = np.zeros(episodes)
for i in range(1):
    step, r, states, states_enemy = ac(1000)
    for i in range(episodes):
        if i == 0:
            timesteps[i] = step[i]
            reward[i] = r[i]
        else:
            timesteps[i] = (timesteps[i] + step[i]) / 2
            reward[i] = reward[i] + r[i]

xaxis = np.asarray([i for i in range(1, 1001)])
plt.title("Waterworld")
plt.ylabel('Steps')
plt.xlabel('Episodes')
plt.plot(xaxis, timesteps)
plt.yscale("log")
plt.show()

xaxis = np.asarray([i for i in range(1, 1001)])
plt.title("Waterworld")
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.plot(xaxis, reward)
plt.show()


window_w = 650
window_h = 500
window = pygame.display.set_mode((window_w, window_h))
white = (255, 255, 255)
black = (0, 0, 255)
FPS = 5
pygame.display.set_caption("Waterworld")
clock = pygame.time.Clock()
block_size = 15
velocity = 3
pygame.init()

for i in range(2 * len(states)):
    x = states[i][0] * 5
    y = states[i][1] * 3

    pos_ex = states_enemy[i][0] * 9
    pos_ey = states_enemy[i][1] * 4

    window.fill(white)
    pygame.draw.rect(window, black, [int((pos_ex)), int((pos_ey)), block_size, block_size])  # creates the enemy
    pygame.draw.circle(window, (140, 40, 40), (int(x), int(y)), 20, 0)  # creates the player
    pygame.draw.circle(window, (40, 50, 0), (483, 300), 20, 0)  # creates the reward creep

    pygame.display.update()
    clock.tick(FPS)
