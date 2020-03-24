"""
Red rectangle:      Explorer
Black rectangle:    Holes   (reward = -1)
Yellow circles:     Goal    (reward = +1)
All other tiles:    Ground  (reward = 0)

The main update loop is in this file. A state s is a tuple of coordinates of the player.
"""

from maze import Maze
from brain import QLearningTable

def update():
    for episode in range(100):
        # initial observation
        s_curr = maze._reset_maze()

        while True:
            maze._render()

            # Get next action from the Q-table
            action = rl.select_next_action(str(s_curr))

            # take the action and observe the next state and reward
            s_next, reward, isDone = maze._update_maze(action)

            # learn from the feedback
            rl.learn(str(s_curr), action, reward, str(s_next))

            s_curr = s_next
            if isDone:
                break

    print("Game over")
    maze.destroy()


if __name__ == "__main__":
    maze = Maze()
    rl = QLearningTable(actions=list(range(len(maze.action_space))))

    maze.after(100, update)
    maze.mainloop()
