# minesweeper_rl.py
import numpy as np
import random
import gym
import time
from gym import spaces
import sys
sys.setrecursionlimit(10000)

class MinesweeperEnv(gym.Env):
    def __init__(self, size=5, n_mines=3):
        super(MinesweeperEnv, self).__init__()
        self.size = size
        self.n_mines = n_mines
        self.action_space = spaces.Discrete(size * size)
        self.observation_space = spaces.Box(low=-2, high=9, shape=(size, size), dtype=np.int8)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.visible = -np.ones((self.size, self.size), dtype=np.int8)
        self.flags = set()
        self.done = False
        self._place_mines()
        return self.visible

    def _place_mines(self):
        mines = random.sample(range(self.size * self.size), self.n_mines)
        self.mines = set()
        for idx in mines:
            x, y = divmod(idx, self.size)
            self.board[x][y] = 9
            self.mines.add((x, y))
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == 9:
                    continue
                self.board[x][y] = self._count_adjacent_mines(x, y)

    def _count_adjacent_mines(self, x, y):
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx][ny] == 9:
                        count += 1
        return count

    def _is_corner(self, x, y):
        corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
        return (x, y) in corners

    def _reveal_recursive(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return
        if self.visible[x][y] != -1:
            return
        self.visible[x][y] = self.board[x][y]
        if self.board[x][y] == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.size and 0 <= ny < self.size):
                        continue
                    self._reveal_recursive(nx, ny)

    def step(self, action):
        x, y = divmod(action, self.size)

        if (x, y) in self.flags:
            return self.visible, -1, self.done, {}

        if self.visible[x][y] != -1:
            return self.visible, -1, self.done, {}
        if (x, y) in self.mines:
            self.visible[x][y] = 9
            self.done = True
            return self.visible, -10, True, {}

        self.visible[x][y] = self.board[x][y]
        if self.board[x][y] == 0:
            self._reveal_recursive(x, y)
        elif self.board[x][y] == 1 and self._is_corner(x, y):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.visible[nx][ny] == -1:
                        self.visible[nx][ny] = self.board[nx][ny]

        if np.count_nonzero(self.visible == -1) == len(self.mines):
            self.done = True
            return self.visible, 20, True, {}
        return self.visible, 2, False, {}

    def flag_tile(self, x, y):
        if self.visible[x][y] == -1:
            self.flags.add((x, y))
            self.visible[x][y] = -2

    def render(self):
        print("Visible Board:")
        for row in self.visible:
            print(" ".join('F' if cell == -2 else str(cell) if cell != -1 else '.' for cell in row))
        print("\n")

class QLearningAgent:
    def __init__(self, env, alpha=0.2, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.env = env
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_state(self, obs):
        return tuple(obs.flatten())

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in range(self.env.action_space.n)}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in range(self.env.action_space.n)}

        predict = self.q_table[state][action]
        target = reward + self.gamma * max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (target - predict)

    def train(self, episodes=10000):
        for ep in range(episodes):
            obs = self.env.reset()
            state = self.get_state(obs)
            done = False
            visited = set()
            max_steps = self.env.size * self.env.size
            steps = 0
            while not done and steps < max_steps:
                available_actions = [
                    a for a in range(self.env.action_space.n)
                    if divmod(a, self.env.size) not in visited and divmod(a, self.env.size) not in self.env.flags
                ]
                if not available_actions:
                    break
                if random.random() < self.epsilon or state not in self.q_table:
                    action = random.choice(available_actions)
                else:
                    action = max(available_actions, key=lambda a: self.q_table[state].get(a, 0))

                x, y = divmod(action, self.env.size)
                if random.random() < 0.01:
                    self.env.flag_tile(x, y)
                    visited.add((x, y))
                    steps += 1
                    continue
                visited.add((x, y))
                next_obs, reward, done, _ = self.env.step(action)
                next_state = self.get_state(next_obs)
                self.learn(state, action, reward, next_state)
                state = next_state
                steps += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if ep % 500 == 0:
                print(f"Episode {ep} complete - Epsilon: {self.epsilon:.3f}")

    def choose_action(self, state, visited):
        possible_actions = [a for a in range(self.env.action_space.n)
                            if divmod(a, self.env.size) not in visited and divmod(a, self.env.size) not in self.env.flags]
        if not possible_actions:
            return self.env.action_space.sample()
        if random.random() < self.epsilon or state not in self.q_table:
            return random.choice(possible_actions)
        return max((a for a in possible_actions), key=lambda a: self.q_table[state].get(a, 0))

    def play(self):
        obs = self.env.reset()
        state = self.get_state(obs)
        done = False
        last_reward = 0
        visited = set()
        while not done:
            action = self.choose_action(state, visited)
            x, y = divmod(action, self.env.size)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            obs, last_reward, done, _ = self.env.step(action)
            state = self.get_state(obs)
        print("Final Board:")
        self.env.render()
        if last_reward > 0:
            print("Game won! 🎉")
        else:
            print("Game lost. 💥")

if __name__ == '__main__':
    env = MinesweeperEnv(size=5, n_mines=3)
    agent = QLearningAgent(env)
    agent.train(episodes=10000)
    print("\nTrained agent playing a game:")
    agent.play()