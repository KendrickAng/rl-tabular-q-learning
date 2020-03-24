import tkinter as tk
import numpy as np
import time

UNIT = 40    #pixels
MAZE_H = 4
MAZE_W = 4

UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3

class Maze(tk.Tk):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = [UP, DOWN, RIGHT, LEFT]
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
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill="red"
        )
        return self.canvas.coords(self.rect)

    def _update_maze(self, action):
        s_curr = self.canvas.coords(self.rect)
        offset = np.array([0, 0])
        if action == UP and s_curr[1] > UNIT:
            offset[1] -= UNIT
        elif action == DOWN and s_curr[1] < (MAZE_H - 1) * UNIT:
            offset[1] += UNIT
        elif action == RIGHT and s_curr[0] < (MAZE_W - 1) * UNIT:
            offset[0] += UNIT
        elif action == LEFT and s_curr[0] > UNIT:
            offset[0] -= UNIT
        self.canvas.move(self.rect, offset[0], offset[1]) # move agent
        s_next = self.canvas.coords(self.rect)

        # next state, reward, maze_done (dead/completed)
        if s_next == self.canvas.coords(self.oval):
            s_next = "TERMINAL"
            return s_next, 1, True
        elif s_next in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            s_next = "TERMINAL"
            return s_next, -1, True

        return s_next, 0, False

    def _render(self):
        time.sleep(0.2)
        self.update()   # tkinter's update() refreshes everything