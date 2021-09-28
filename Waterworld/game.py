import numpy as np


class Player:
    def __init__(self, height, width, velocity):
        self.width = 50
        self.height = 50
        self.actions = ["up", "down", "left", "right"]
        self.velocity = velocity
        self.state_ub = [100, 100]
        self.state_lb = [0, 0]
        self.vel = 0.25*width

    def movement(self, pos_x, pos_y, act, enemy_x, enemy_y):
        terminate = False
        goal_reached = False
        if act == "up":
            pos_x = pos_x + self.velocity

        if act == "down":
            pos_x = pos_x - self.velocity

        if act == "left":
            pos_y = pos_y - self.velocity

        if act == "right":
            pos_y = pos_y + self.velocity

        pos_x = min(max(pos_x, self.state_lb[0]), self.state_ub[0])
        pos_y = min(max(pos_y, self.state_lb[1]), self.state_ub[1])

        if pos_x == self.state_ub[0]:
            if pos_y == self.state_ub[1]:
                goal_reached = True

        new_enemy_x, new_enemy_y = self.enemy_update(enemy_x, enemy_y)

        if pos_x == new_enemy_x and pos_y == new_enemy_y:
            goal_reached = True
            terminate = True

        reward = self.reward_function(pos_x, pos_y, new_enemy_x, new_enemy_y, goal_reached, terminate)

        return pos_x, pos_y, goal_reached, new_enemy_x, new_enemy_y, reward

    def enemy_update(self, enemy_x, enemy_y):
        block_size = 1.5
        enemy_x += self.vel
        if enemy_x + block_size > self.width or enemy_x < 0:
            self.vel = -self.vel
        return enemy_x, enemy_y

    def reward_function(self, pos_x, pos_y, enemy_x, enemy_y, goal_reached, terminate):
        r = 0
        if goal_reached == True:
            if terminate == True:
                r = -1000
            else:
                r = 1000
        else:
            r = 1
            posit = np.array((pos_x, pos_y))
            enemy = np.array((enemy_x, enemy_y))
            temp = posit - enemy
            sum = np.dot(temp.T, temp)
            dist = np.sqrt(sum)
            if dist < 5:
                r = r - 200

        return r


