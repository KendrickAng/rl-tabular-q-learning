"""
Red rectangle:      Explorer
Black rectangle:    Holes   (reward = -1)
Yellow circles:     Goal    (reward = +1)
All other tiles:    Ground  (reward = 0)

The main update loop is in this file.
"""

from maze import Maze
from brain import QLearningTable

def main():
    maze = Maze()
    # maze.after(100, update)
    maze.mainloop()
    # rl = QLearningTable(actions=list(len(maze.action_space))

if __name__ == "__main__":
    main()