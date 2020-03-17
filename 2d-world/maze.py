import tkinter as tk
import numpy as np
import time

UNIT = 40    #pixels
MAZE_H = 4
MAZE_W = 4

class Maze(tk.Tk):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.title('maze')
        self.geometry("{0}x{1}".format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._init_maze()

    def _init_maze(self):
        self.canvas = tk.Canvas(self, bg="white", height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT # vertical lines
            self.canvas.create_line((x0, y0, x1, y1))
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r # horizontal lines
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # black holes are hardcoded
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        # create rectangles by specifying the points of bottom-left and top-right corners
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create goal oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create player red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()


    def _reset_maze(self):
        pass

    def _update_maze(self):
        pass

    def _render(self):
        time.sleep(0.1)
        self.update()   # tkinter's update() refreshes everything