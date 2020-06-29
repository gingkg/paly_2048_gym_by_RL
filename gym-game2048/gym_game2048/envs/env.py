import gym
import numpy as np
import sys
from gym.utils import seeding
from gym import spaces
from .game_2048 import Game2048
from tkinter import *
import time


SIZE = 400
GRID_LEN = 4
GRID_PADDING = 6

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}

CELL_COLOR_DICT = {
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
    32: "#f9f6f2",
    64: "#f9f6f2",
    128: "#f9f6f2",
    256: "#f9f6f2",
    512: "#f9f6f2",
    1024: "#f9f6f2",
    2048: "#f9f6f2",
}

FONT = ("Fira Code", 30, "bold")


class GameGrid(Frame):
    def __init__(self, episode_log):
        """
        Implementing game grid.

        Show a Tkinter window with the game board.

        Parameters
        ----------
        boards : List
            List of boards of a played game.
        """
        Frame.__init__(self)

        self.grid()
        self.master.title("2048    action: {}    socre: {}".format("#", 0))
        self.boards = episode_log["boards"]
        self.actions = episode_log["actions"]
        self.scores = episode_log["scores"]

        self.matrix = self.boards[0]
        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()

        self.wait_visibility()
        self.after(10, self.make_move())

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(
                    master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2
                )
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def make_move(self):
        for board, action, score in zip(self.boards, self.actions, self.scores):
            time.sleep(0.1)
            self.master.title("2048    action: {}    score: {}".format(action, score))
            self.matrix = board
            self.update_grid_cells()
        time.sleep(2)
        # sys.exit()


class Game2048Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, board_size, binary, extractor, invalid_move_warmup=16, invalid_move_threshold=0.1, penalty=-512, seed=None
    ):
        """
        Iniatialize the environment.


        Parameters
        ----------
        board_size : int
            Size of the board. Default=4
        binary : bool
            Use binary representation of the board(power 2 matrix). Default=True
        extractor : str
            Type of model to extract the features. Default=cnn
        invalid_move_warmup : int
            Minimum of invalid movements to finish the game. Default=16
        invalid_move_threshold : float
                    How much(fraction) invalid movements is necessary according to the total of moviments already executed. to finish the episode after invalid_move_warmup. Default 0.1
        penalty : int
            Penalization score of invalid movements to sum up in reward function. Default=-512
        seed :  int
            Seed
        """

        self.state = np.zeros(board_size * board_size)
        self.__binary = binary
        self.__extractor = extractor

        if self.__binary is True:
            if extractor == "cnn":
                self.observation_space = spaces.Box(
                    0, 1, (board_size, board_size, 16 + (board_size - 4)), dtype=np.uint32
                )
            else:
                self.observation_space = spaces.Box(
                    0, 1, (board_size * board_size * (16 + (board_size - 4)),), dtype=np.uint32
                )
        else:
            if extractor == "mlp":
                self.observation_space = spaces.Box(0, 2 ** 16, (board_size * board_size,), dtype=np.uint32)
            else:
                ValueError("Extractor must to be mlp when observation space is not binary")

        self.action_space = spaces.Discrete(4)  # Up, down, right, left

        if penalty > 0:
            raise ValueError("The value of penalty needs to be between [0, -inf)")
        self.__game = Game2048(board_size, invalid_move_warmup, invalid_move_threshold, penalty)
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__board_size = board_size

    def step(self, action):
        """
        Execute a action.

        Parameters
        ----------
        action : int
            Action selected by the model.
        """
        reward = 0
        info = dict()

        before_move = self.__game.get_board().copy()
        self.__game.make_move(action)
        self.__game.confirm_move()

        if self.__binary is True:
            self.__game.transform_board_to_power_2_mat()

            if self.__extractor == "cnn":
                self.state = self.__game.get_power_2_mat()
            else:
                self.state = self.__game.get_power_2_mat().flatten()
        else:
            self.state = self.__game.get_board().flatten()

        self.__done, penalty = self.__game.verify_game_state()
        reward = self.__game.get_move_score() + penalty
        self.__n_iter = self.__n_iter + 1
        after_move = self.__game.get_board()

        info["total_score"] = self.__game.get_total_score()
        info["steps_duration"] = self.__n_iter
        info["before_move"] = before_move
        info["after_move"] = after_move

        return (self.state, reward, self.__done, info)

    def reset(self):
        "Reset the environment."
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__game.reset()
        if self.__binary is True:
            self.__game.transform_board_to_power_2_mat()
            if self.__extractor == "cnn":
                self.state = self.__game.get_power_2_mat()
            else:
                self.state = self.__game.get_power_2_mat().flatten()
        else:
            self.state = self.__game.get_board().flatten()
        return self.state

    def render(self, episode_log=None, mode="human"):
        root = Tk()
        GameGrid(episode_log)
        print("渲染完毕！")

    def get_board(self):
        "Get the board."
        return self.__game.get_board()

