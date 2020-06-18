from game2048.agents import Agent
from models.Final_Model import RCNN_model
import numpy as np

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

class MyAgent(Agent):
    
    def step(self):
        
        board = np.array([reshape_board(self.game.board)])
        prediction = model.predict(board)
        direction = np.argmax(prediction, axis = 1)
        return int(direction)