
# coding: utf-8

# In[1]:


import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Intellecto:
    def __init__(self):
        self.NONE_MOVE_SCORE = -100000 * (9**3)
        self.DEPTH = 1
        self.n_bubbles = 5
        self.queue_size = 30
        self.min_bubble_val = -9
        self.max_bubble_val = 9

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


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


    def play_move(self, i, board, queue, dry_simulation=False):
        valid_move = True
        if self.is_game_over(board) or i < 0 or i >= len(board) or board[i] is None:
            valid_move = False
            return valid_move, board, queue, None

        move_score = self.get_move_score(i, board, queue, dry_simulation=dry_simulation)
        next_board, next_queue = self.get_next_board_state(i, board, queue)
        return valid_move, next_board, next_queue, move_score


    def play_game(self, board, queue):
        board_queue_moves_score_map = []
        game_over = False
        game_score = 0
        while not game_over:
            raw_moves_scores = []
            for i in range(len(board)):
                valid_move, _, _, move_score = self.play_move(i, board, queue)
                if not valid_move:
                    raw_moves_scores.append(self.NONE_MOVE_SCORE)
                else:
                    raw_moves_scores.append(move_score)

            moves_score = self.softmax(raw_moves_scores)
            board_queue_moves_score_map.append({"board":board, "queue":queue, "moves_score":moves_score})
            valid_move = False
            while not valid_move:
                i = np.random.choice(len(moves_score), p=moves_score)
                valid_move, board, queue, raw_move_score = self.play_move(i, board, queue)

            game_score += raw_move_score
            game_over = self.is_game_over(board)

        return board_queue_moves_score_map, game_score

    def play_episode(self, n_games=25, queue_size=None):
        if queue_size is None: queue_size = self.queue_size
        x_episode = []
        y_episode = []

        for n_game in range(n_games):
            x_game = []
            y_game = []

            board, queue = self.reset_game(queue_size=queue_size)
            game_info_map, game_score = self.play_game(board=board, queue=queue)
            for info in game_info_map:
                b = info['board']
                q = info['queue']
                s = info['moves_score']

                if len(q) > 0:
                    b.append(q[0])
                else:
                    b.append(None)

                x = []
                for i in range(len(b)):
                    f = np.zeros((len(b) + 1), dtype=np.float64)
                    if b[i] is not None:
                        f[i] = 1
                        f[len(b)] = b[i]

                    x.append(f)

                x = np.array(x, dtype=np.float64)
                x = np.reshape(x, (1, x.size))
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
