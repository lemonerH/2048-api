import numpy as np
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent


def board2array(game):
    vec = np.zeros((4, 4, 16), dtype = bool)
    for i in range(4):
        for j in range(4):
	        k = int(game.board[i, j])
	        if k != 0:
		        k = int(np.log2(k))
        	vec[i][j][k] = 1
    return vec

def step2array(step):
    vec = np.zeros(4, dtype = bool)
    vec[step] = 1
    return vec

def data_generator_for_CNN(score_to_begin, score_to_win, batch_size):
    datas = []
    labels = []
    cnt = 0
    while 1:
        game = Game(score_to_win = score_to_win, random = False)
        agent = ExpectiMaxAgent(game)
        while game.end == 0:
            step = agent.step()
            if game.score >= score_to_begin:
                datas.append(board2array(game))
                labels.append(step2array(step))
                cnt += 1
            game.move(step)
            if cnt == batch_size:
                cnt = 0
                datas = np.array(datas)
                labels = np.array(labels)
                yield (datas, labels)
                datas = []
                labels = []

def data_generator_for_RNN(score_to_begin, score_to_win, batch_size):
    datas = []
    labels = []
    cnt = 0
    while 1:
        game = Game(score_to_win = score_to_win, random = False)
        agent = ExpectiMaxAgent(game)
        while game.end == 0:
            step = agent.step()
            if game.score >= score_to_begin:
                datas.append(board2array(game).reshape(256))
                labels.append(step2array(step))
                cnt += 1
            game.move(step)
            if cnt == batch_size:
                cnt = 0
                datas = np.array(datas)
                labels = np.array(labels)
                yield (datas, labels)
                datas = []
                labels = []

def data_generator_for_CRNN(score_to_begin, score_to_win, batch_size):
    datas = []
    labels = []
    cnt = 0
    while 1:
        game = Game(score_to_win = score_to_win, random = False)
        agent = ExpectiMaxAgent(game)
        while game.end == 0:
            step = agent.step()
            if game.score >= score_to_begin:
                board = board2array(game)
                board1 = np.swapaxes(board, 1, 2)
                board2 = np.swapaxes(board1, 0, 1).reshape((16, 4, 4, 1))
                
                datas.append(board2)
                labels.append(step2array(step))
                cnt += 1
            game.move(step)
            if cnt == batch_size:
                cnt = 0
                datas = np.array(datas)
                labels = np.array(labels)
                yield (datas, labels)
                datas = []
                labels = []


def data_generator(batch_size):
    datas = []
    labels = []
    cnt = 0
    while 1:
        game = Game(score_to_win = 2048, random = False)
        agent = ExpectiMaxAgent(game)
        while game.end == 0:
            step = agent.step()
            board = game.board / 11
            board1 = board.T
            datas.append(np.hstack((board, board1)))
            labels.append(step2array(step))
            cnt += 1
            game.move(step)
            if cnt == batch_size:
                cnt = 0
                datas = np.array(datas)
                labels = np.array(labels)
                yield (datas, labels)
                datas = []
                labels = []
