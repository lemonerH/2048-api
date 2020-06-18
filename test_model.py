from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent
import numpy as np
from models.Final_Model import RCNN_model

model = RCNN_model()

model.load_weights("checkpoints/checkpoint.hdf5")

def reshape_board(board):
    res = np.zeros((4, 4), dtype = float)
    for i in range(4):
        for j in range(4):
            k = int(board[i, j])
            if k != 0:
                res[i, j] = np.log2(k) / 11

    res1 = res.T
    return np.hstack((res, res1))


game3 = Game(score_to_win = 2048, random = False)
display3 = Display()

while game3.end == 0:
    display3.display(game3)
    # agent1 = ExpectiMaxAgent(game3)
    board = np.array([reshape_board(game3.board)])
    prediction = model.predict(board)
    step = np.argmax(prediction, axis = 1)
    # step = agent1.step()

    game3.move(step)