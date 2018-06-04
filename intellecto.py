
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import extmath
import warnings
warnings.filterwarnings("ignore")

class Intellecto:
    def __init__(self):
        self.NONE_MOVE_SCORE = -2 * (9**3)
        self.DEPTH = 1
        self.n_bubbles = 5
        self.queue_size = 30
        self.min_bubble_val = -9
        self.max_bubble_val = 9
        self.n_diificulties = 5


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def sigmoid(self, x):
        x_ = np.multiply(x, -100)
        return 1 / (1 + np.exp(x_))


    def squashed_score(self, x):
        return np.divide(x, 10)

    def one_hot_scores(self, x):
        x_oh = np.zeros(len(x), dtype=np.float64)
        x_oh[np.argmax(x)] = 1
        return x_oh


    def reset_game(self, queue_size=None):
        if queue_size is None: queue_size = self.queue_size
        board = np.random.random_integers(low=self.min_bubble_val, high=self.max_bubble_val, size=self.n_bubbles).tolist()
        queue = np.random.random_integers(low=self.min_bubble_val, high=self.max_bubble_val, size=queue_size).tolist()
        return board, queue


    def get_left_val(self, i, board):
        if i == 0: return 1
        left_i = i - 1;
        while left_i >= 0:
            if board[left_i] is None:
                left_i -= 1
                continue
            else: break

        if left_i >= 0: return board[left_i]
        else: return 1


    def get_right_val(self, i, board):
        if i == (len(board) - 1): return 1
        right_i = i + 1;
        while right_i < len(board):
            if board[right_i] is None:
                right_i += 1
                continue
            else: break

        if right_i < len(board): return board[right_i]
        else: return 1


    def get_next_board_state(self, i, board, queue):
        board_copy = board[:]
        queue_copy = queue[:]
        replacement_val = None
        if len(queue_copy) > 0:
            replacement_val = queue_copy.pop(0)

        board_copy[i] = replacement_val
        return board_copy, queue_copy


    def get_deep_move_score(self, board, queue, depth):
        if depth > self.DEPTH: return 0

        val_scores = []
        deep_val_scores = []

        for i in range(len(board)):
            if board[i] is None: continue

            left_val = self.get_left_val(i, board)
            right_val = self.get_right_val(i, board)
            val = left_val * board[i] * right_val
            val_scores.append(val)

            next_board, next_queue = self.get_next_board_state(i, board, queue)
            deep_val = self.get_deep_move_score(next_board, next_queue, depth + 1)
            deep_val_scores.append(deep_val)

        deep_score = self.NONE_MOVE_SCORE
        for i in range(len(val_scores)):
            if (val_scores[i] - deep_val_scores[i]) > deep_score:
                deep_score = val_scores[i] - deep_val_scores[i]

        return deep_score


    def get_move_score(self, i, board, queue, dry_simulation=False):
        if board[i] is None: return self.NONE_MOVE_SCORE

        left_val = self.get_left_val(i, board)
        right_val = self.get_right_val(i, board)
        val = left_val * board[i] * right_val
        deep_val = 0

        if self.DEPTH > 0 and not dry_simulation:
            next_board, next_queue = self.get_next_board_state(i, board, queue)
            if not self.is_game_over(next_board):
                deep_val = self.get_deep_move_score(next_board, next_queue, 1)

        return val - deep_val


    def is_game_over(self, board):
        return sum(np.array(board) != None) == 0


    def is_move_valid(self, i, board):
        if self.is_game_over(board) or i < 0 or i >= len(board) or board[i] is None:
            return False

        return True


    def play_move(self, i, board, queue, dry_simulation=False):
        valid_move = self.is_move_valid(i, board)
        if not valid_move:
            return valid_move, board, queue, None

        move_score = self.get_move_score(i, board, queue, dry_simulation=dry_simulation)
        next_board, next_queue = self.get_next_board_state(i, board, queue)
        return valid_move, next_board, next_queue, move_score

    def choose_a_move(self, board, raw_moves_scores, difficulty=None):

        if difficulty is None or difficulty > 4:
            difficulty = 0

        ordering = np.argsort(-1 * np.array(raw_moves_scores))
        valid_ordering = []
        n_valid_options = 0.0
        for option in ordering:
            valid_move = self.is_move_valid(option, board)
            if valid_move:
                n_valid_options += 1
                valid_ordering.append(option)

        mapping_ratio = n_valid_options / self.n_bubbles
        mapped_difficulty = np.int(np.floor(mapping_ratio * difficulty))
        return valid_ordering[mapped_difficulty]

    def play_a_move_on_board(self, board, queue, difficulty=None):
        raw_moves_scores = []
        for i in range(len(board)):
            valid_move, _, _, move_score = self.play_move(i, board, queue)
            if not valid_move:
                raw_moves_scores.append(self.NONE_MOVE_SCORE)
            else:
                raw_moves_scores.append(move_score)

        #moves_score = self.softmax(raw_moves_scores)
        moves_score = self.one_hot_scores(raw_moves_scores)
        #moves_score = self.sigmoid(raw_moves_scores)
        #moves_score = self.squashed_score(raw_moves_scores)

        i = self.choose_a_move(board, raw_moves_scores, difficulty)
        valid_move, board_, queue_, raw_move_score = self.play_move(i, board, queue)
        return i, board_, queue_, moves_score, raw_move_score, raw_moves_scores


    def play_game(self, board, queue, difficulty=None):
        board_queue_moves_score_map = []
        game_over = False
        game_score = 0
        n_moves = 0
        while not game_over:
            move, board_, queue_, moves_score, raw_move_score, raw_moves_score = self.play_a_move_on_board(board=board, queue=queue, difficulty=difficulty)
            board = board_
            queue = queue_
            board_queue_moves_score_map.append({"board": board, "queue": queue, "moves_score": moves_score})
            game_score += raw_move_score
            n_moves += 1
            game_over = self.is_game_over(board)

        return board_queue_moves_score_map, game_score

    def one_hot_board(self, b, q):
        board = b[:]
        queue = q[:]
        if len(queue) > 0: board.append(queue[0])
        else: board.append(None)

        x = []
        for i in range(len(board)):
            f = np.zeros((len(board) + 1), dtype=np.float64)
            if board[i] is not None:
                f[i] = 1
                f[len(board)] = board[i]

            x.append(f)

        x = np.array(x, dtype=np.float64)
        x = np.reshape(x, (1, x.size))
        return x

    def play_episode(self, n_games=25, queue_size=None, difficulty=None):
        if queue_size is None: queue_size = self.queue_size
        x_episode = []
        y_episode = []

        for n_game in range(n_games):
            x_game = []
            y_game = []

            board, queue = self.reset_game(queue_size=queue_size)
            game_info_map, game_score = self.play_game(board=board, queue=queue, difficulty=difficulty)
            for info in game_info_map:
                b = info['board']
                q = info['queue']
                s = info['moves_score']

                x = self.one_hot_board(b=b, q=q)
                y = np.reshape(s, (1, s.size))

                if len(x_game) == 0:
                    x_game = x
                    y_game = y
                else:
                    x_game = np.concatenate((x_game, x), axis=0)
                    y_game = np.concatenate((y_game, y), axis=0)

            if n_game == 0:
                x_episode = x_game
                y_episode = y_game
            else:
                x_episode = np.concatenate((x_episode, x_game), axis=0)
                y_episode = np.concatenate((y_episode, y_game), axis=0)

        return x_episode, y_episode

    def plot(self, y_data, y_label, window=1, windowshift=1):
        filename = y_label + "_record.pdf"
        y_data_mean = [0]
        index = 0
        while True:
            if index > (len(y_data) - window):
                break

            fr = np.int(index)
            to = np.int(index + window)
            w = y_data[fr:to]
            y_data_mean.append(sum(w) * 1.0 / window)
            index = index + windowshift

        x_data = [(x+1) for x in range(len(y_data_mean))]
        plt.plot(x_data, y_data_mean, linewidth=1)
        plt.xlabel('Simulations')
        plt.ylabel(y_label)
        plt.savefig(filename)