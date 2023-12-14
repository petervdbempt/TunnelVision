import numpy as np


class TunnelVision:
    def __init__(self, env):
        if env == 'Standard':
            self.rows = 4
            self.cols = 12
        self.grid = np.zeros((self.rows, self.cols), dtype=int)  # 0 represents empty cells
        self.agent_position = (0, 0)  # Agent starts in the top-left corner
        self.goal_position = (self.rows - 1, self.cols - 1)  # Goal is in the bottom-right corner
        self.terminated = False
        self.truncated = False
        self.num_states = self.rows * self.cols
        self.num_actions = 4
        self.build_grid()

    def build_grid(self):
        # Build environment (1 represents toxic gas, 2 represents inferior reward, 3 represents goal)
        self.grid[0, 3] = 1
        self.grid[1, 3] = 1
        self.grid[3, 3] = 1

        self.grid[0, 7] = 1
        self.grid[2, 7] = 1
        self.grid[3, 7] = 1

        self.grid[0, 11] = 1
        self.grid[1, 11] = 1
        self.grid[2, 11] = 1

        self.grid[3, 1] = 2
        self.grid[0, 6] = 2
        self.grid[3, 11] = 3

    def print_grid(self):
        for row in self.grid:
            print(" ".join(map(str, row)))
        print()

    def coordinates_to_index(self, row, col):
        # Convert (row, col) coordinates to a one-dimensional index
        return row * self.cols + col

    def step(self, action):
        # Actions: 0 - up, 1 - right, 2 - down, 3 - left
        new_position = self.agent_position
        reward = 0

        if action == 0 and self.agent_position[0] > 0:
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1 and self.agent_position[1] < self.cols - 1:
            new_position = (self.agent_position[0], self.agent_position[1] + 1)
        elif action == 2 and self.agent_position[0] < self.rows - 1:
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 3 and self.agent_position[1] > 0:
            new_position = (self.agent_position[0], self.agent_position[1] - 1)

        # Check for events
        if self.grid[new_position] == 1:
            # print("Agent encountered toxic gas! Game Over.")
            self.terminated = True

        # Check for rewards
        elif self.grid[new_position] == 2:
            # print("Agent found an inferior reward.")
            reward = 0.25
            self.terminated = True

        # Check for goal
        elif self.grid[new_position] == 3:
            # print("Goal reached!")
            reward = 1
            self.terminated = True

        # Update agent position
        self.agent_position = new_position
        state_index = self.coordinates_to_index(*self.agent_position)
        # print(self.agent_position)
        # self.print_grid()

        return state_index, reward, self.terminated, self.truncated

    def reset(self):
        self.agent_position = (0, 0)  # Agent starts in the top-left corner
        self.terminated = False
        self.truncated = False
