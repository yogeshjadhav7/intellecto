
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import extmath
from sklearn.preprocessing import PolynomialFeatures
from keras.models import load_model
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")

class Intellecto:
    def __init__(self):
        self.NONE_MOVE_SCORE = -1.0 * (9 ** 3)
        self.DEPTH = 1
        self.n_bubbles = 5
        self.queue_size = 30
        self.min_bubble_val = -9
        self.max_bubble_val = 9
        self.n_diificulties = 5
        self.poly = PolynomialFeatures(degree=3, include_bias=False)

    def _build_model_path(self, userId, dir="intellecto_service/saved_models/", one_hot=True):
        model_name = str(userId)
        if one_hot: model_name = model_name + ".pkl"
        else: model_name = model_name + ".hdf5"
        return dir + model_name


    def get_saved_model(self, userId, dir="intellecto_service/saved_models/", one_hot=True):
        model_path = self._build_model_path(userId=userId, dir=dir, one_hot=one_hot)
        if one_hot: return joblib.load(model_path)
        else: return load_model(model_path)


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


    def categorize(self, x):
        return np.array([np.argmax(x)], dtype=np.int)


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

        deep_score = None
        for i in range(len(val_scores)):
            if deep_score is None:
                deep_score = val_scores[i] - deep_val_scores[i]
                continue

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

    def validity_of_moves(self, board):
        validity = []
        for i in range(len(board)): validity.append(self.is_move_valid(i=i, board=board))
        return np.array([validity]) * 1

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
        #print("raw_moves_scores", raw_moves_scores, ", valid_ordering", valid_ordering, ", mapping_ratio", mapping_ratio, ", mapped_difficulty", mapped_difficulty)
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
        #moves_score = self.one_hot_scores(raw_moves_scores)
        #moves_score = self.sigmoid(raw_moves_scores)
        #moves_score = self.squashed_score(raw_moves_scores)
        #moves_score = self.categorize(raw_moves_scores)

        i = self.choose_a_move(board, raw_moves_scores, difficulty)
        valid_move, board_, queue_, raw_move_score = self.play_move(i, board, queue)
        moves_score = [0 for x in range(self.n_bubbles)]
        moves_score[i] = 1
        moves_score = np.array(moves_score)
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
            board_queue_moves_score_map.append({"board": board, "queue": queue, "moves_score": moves_score, "move": move,
                                                "raw_moves_score": raw_moves_score})
            game_score += raw_move_score
            n_moves += 1
            game_over = self.is_game_over(board)

        return board_queue_moves_score_map, game_score

    def calculate_relative_raw_moves_score(self, raw_moves_score):
        raw_moves_score = np.subtract(raw_moves_score, np.amin(raw_moves_score))
        raw_moves_score = np.divide(raw_moves_score, 0.001 + np.amax(raw_moves_score))
        return raw_moves_score

    def prediction_method(self, model, x, one_hot=True):
        #if one_hot: return model.predict_proba(x)
        return model.predict(x)

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
                f[len(board)] = board[i] * 0.1

            x.append(f)

        x = np.array(x, dtype=np.float64)
        x = np.reshape(x, (1, x.size))
        return x

    def parse_game_state_str(self, game_state_str):
        game_state = game_state_str.split(',')
        size = len(game_state)
        itr = 0
        b = []
        q = []
        while itr < size:
            if game_state[itr] == '#':
                val = None
            else:
                val = int(game_state[itr])
            if itr < 5:
                b.append(val)
            else:
                q.append(val)
            itr += 1

        return b, q

    def _get_cv_parameters_range(self, n_trees, prev_best_parameters=None, prev_parameters_config=None):
        params = ['max_depth', 'min_samples_split', 'min_samples_leaf']
        parameters_range = {}
        if prev_best_parameters is None:
            parameters_range[params[0]] = [1, 4 * n_trees]
            parameters_range[params[1]] = [2, 4 * n_trees]
            parameters_range[params[2]] = [1, 4 * n_trees]

        else:
            for param in params:
                best_parameter_value = prev_best_parameters[param]
                prev_parameter_config = prev_parameters_config[param]
                index = prev_parameter_config.index(best_parameter_value)
                size = len(prev_parameter_config)
                if size == 1:
                    parameters_range[param] = [best_parameter_value, best_parameter_value]
                    continue

                if index == 0:
                    start = best_parameter_value
                    end = prev_parameter_config[index + 1] - 1
                    if start >= end: end = start
                    parameters_range[param] = [start, end]
                    continue

                if index == size - 1:
                    end = best_parameter_value
                    start = prev_parameter_config[index - 1] + 1
                    if start >= end: start = end
                    parameters_range[param] = [start, end]
                    continue

                start = prev_parameter_config[index - 1] + 1
                end = prev_parameter_config[index + 1] - 1
                if start >= end:
                    start = best_parameter_value
                    end = best_parameter_value

                parameters_range[param] = [start, end]

        return params, parameters_range

    def cross_validate_and_fit(self, x, y, verbose=0):
        count = len(x)
        base_n_trees = 64 #128
        n_trees = 2 ** (int(np.log2(count + 1) / np.log2(6)))
        n_cv = max(2, int(count / (100)))

        if verbose == 1:
            print("base_n_trees", base_n_trees)
            print("n_trees", n_trees)
            print("n_cv", n_cv)
            print("count", count)

        rf_clf = None
        prev_best_parameters = None
        prev_parameters_config = None

        while True:
            terminate = True
            params, parameters_range = self._get_cv_parameters_range(n_trees=n_trees, prev_best_parameters=prev_best_parameters,
                                                            prev_parameters_config=prev_parameters_config)

            for param in params:
                parameter_range = parameters_range[param]
                if parameter_range[0] != parameter_range[1]: terminate = False

            if terminate: break

            if verbose == 1:
                print("\n\n")
                print("parameters_range", parameters_range)

            parameters_config = {}
            for param in params:
                parameter_range = parameters_range[param]
                if prev_best_parameters is not None:
                    best_parameter_val = prev_best_parameters[param]
                else:
                    best_parameter_val = None
                best_parameter_added = False
                start = parameter_range[0]
                end = parameter_range[1]
                val = [start]
                if start < end:
                    c = 0
                    while True:
                        v = start + (2 ** c)
                        if v >= end: break
                        if best_parameter_val is not None and best_parameter_val == v: best_parameter_added = True
                        if best_parameter_val is not None and best_parameter_val < v and not best_parameter_added:
                            val.append(best_parameter_val)
                            best_parameter_added = True

                        val.append(v)
                        c += 1

                    val.append(end)

                parameters_config[param] = val

            if verbose == 1: print("parameters_config", parameters_config)

            clf = GridSearchCV(estimator=RandomForestRegressor(random_state=42,
                                                                n_estimators=base_n_trees + n_trees,
                                                                oob_score=True),
                               param_grid=parameters_config,
                               cv=n_cv,
                               verbose=verbose,
                               n_jobs=8,
                               scoring='mean_squared_error')

            clf.fit(x, y)
            prev_best_parameters = clf.best_params_
            prev_parameters_config = parameters_config
            rf_clf = clf.best_estimator_
            clf = None

        print("Best Parameters: ", prev_best_parameters)
        print("Training Score: ", rf_clf.score(x, y))
        return rf_clf, prev_best_parameters


    def save_model(self, model, userId, dir="intellecto_service/saved_models/"):
        if userId is None or dir is None or model is None: return None
        model_path = self._build_model_path(userId=userId, dir=dir)

        try:
            joblib.dump(model, model_path, protocol=2)
            return model_path
        except:
            return None


    def load_model(self, model_path):
        try: return joblib.load(model_path)
        except: return None


    def augment_training_data(self, features, labels, one_hot=True):
        size = len(features)
        f_set = []
        for f in features:
            f_set.append(str(f))

        for index in range(size):
            x = features[index]
            y = labels[index]

            augmented_x = np.flip(x, axis=0)
            augmented_y = y

            if str(augmented_x) in f_set: continue

            if not one_hot:
                response = np.argmax(augmented_y)
                augmented_y[response] = 0
                response = self.n_bubbles - 1 - response
                augmented_y[response] = 1

            augmented_x = np.array([augmented_x])
            augmented_y = np.array([augmented_y])

            features = np.concatenate((features, augmented_x), axis=0)
            labels = np.concatenate((labels, augmented_y), axis=0)

        return features, labels


    def train_model(self, features, labels, verbose=0, one_hot=True):
        #features, labels = self.augment_training_data(features=features, labels=labels, one_hot=one_hot)
        model, _ = self.cross_validate_and_fit(x=features, y=labels, verbose=verbose)
        return model


    def train(self, behaviourList, one_hot=True, verbose=0, auto_correct_ratio=0.1, auto_correction_degree=0.5):
        X = []
        Y = []
        info_list = []

        for behaviour in behaviourList:
            index = len(X)
            game_state_str = behaviour['gameState']
            b, q = self.parse_game_state_str(game_state_str=game_state_str)
            x = self.one_hot_board(b=b, q=q)
            user_response = behaviour['userResponse']
            correct_response, _, _, _, _, raw_moves_scores = self.play_a_move_on_board(board=b, queue=q)
            relative_scores = self.calculate_relative_raw_moves_score(raw_moves_score=raw_moves_scores)

            if not one_hot:
                y = np.zeros((1,self.n_bubbles), dtype=np.int)
                y[user_response] = 1
            else:
                y = np.zeros((1,1), dtype=np.float64)
                y[0] = relative_scores[user_response]

            if len(X) == 0:
                X = x
                Y = y
            else:
                X = np.concatenate((X, x), axis=0)
                Y = np.concatenate((Y, y), axis=0)

            info_element = {"index": index, "user_response": user_response,"relative_scores": relative_scores,
                            "diff": np.abs(relative_scores[user_response] - relative_scores[correct_response])}

            info_list.append(info_element)

        n_auto_correct = int(len(behaviourList) * auto_correct_ratio)
        print("n_auto_correct", n_auto_correct)
        info_list = sorted(info_list, key=itemgetter('diff'), reverse=True)
        for info_element in info_list:
            if n_auto_correct <= 0: break
            relative_scores = info_element["relative_scores"]
            user_response = info_element["user_response"]
            diff = info_element["diff"]
            index = info_element["index"]
            preds_score = relative_scores[user_response]
            preds_score += diff * auto_correction_degree

            if not one_hot:
                preds_score = 1 - np.abs(np.subtract(relative_scores, preds_score[:, None]))
                response = np.argmax(preds_score, axis=1).flatten()
                Y[index][user_response] = 0
                Y[index][response] = 1
            else:
                Y[index][0] = preds_score

            n_auto_correct -= 1

        model = self.train_model(features=X, labels=Y, verbose=verbose, one_hot=one_hot)
        return model

    def predict(self, behaviourMap, saved_model, one_hot=True):
        game_state_strs = list(behaviourMap.keys())
        size = len(game_state_strs)
        X = []
        X_valid = []
        relative_moves_scores = []

        for game_state_str in game_state_strs:
            b, q = self.parse_game_state_str(game_state_str=game_state_str)

            _, _, _, _, _, raw_moves_scores = self.play_a_move_on_board(board=b, queue=q)
            relative_scores = self.calculate_relative_raw_moves_score(raw_moves_score=raw_moves_scores)
            relative_moves_scores.append(relative_scores)

            x = self.one_hot_board(b=b, q=q)
            if len(X) == 0: X = x
            else: X = np.concatenate((X, x), axis=0)

            x_valid = self.validity_of_moves(board=b)
            if len(X_valid) == 0: X_valid = x_valid
            else: X_valid = np.concatenate((X_valid, x_valid), axis=0)

        preds_score = self.prediction_method(model=saved_model, x=X, one_hot=one_hot)

        if one_hot: preds_score = 1 - np.abs(np.subtract(relative_moves_scores, preds_score[:, None]))

        preds_score = np.multiply(preds_score, X_valid)
        preds = np.argmax(preds_score, axis=1).flatten()

        cnt = 0
        while cnt < size:
            key = game_state_strs[cnt]
            behaviour = behaviourMap[key]
            behaviour['robotResponse'] = preds[cnt]
            behaviourMap[key] = behaviour
            cnt += 1


    def play_episode(self, n_games=25, queue_size=None, difficulty=None):
        if queue_size is None: queue_size = self.queue_size
        x_episode = []
        y_episode = []
        b_episode = []
        q_episode = []
        raw_moves_score_episode = []

        for n_game in range(n_games):
            x_game = []
            y_game = []
            raw_moves_score_game = []

            board, queue = self.reset_game(queue_size=queue_size)
            game_info_map, game_score = self.play_game(board=board, queue=queue, difficulty=difficulty)
            for info in game_info_map:
                b = info['board']
                q = info['queue']
                s = info['moves_score']
                r_s = info["raw_moves_score"]

                x = self.one_hot_board(b=b, q=q)
                y = np.reshape(s, (1, s.size))

                if len(x_game) == 0:
                    x_game = x
                    y_game = y
                    raw_moves_score_game = np.array([r_s]) * 1.0
                    b_episode.append(b)
                    q_episode.append(q)
                else:
                    x_game = np.concatenate((x_game, x), axis=0)
                    y_game = np.concatenate((y_game, y), axis=0)
                    b_episode.append(b)
                    q_episode.append(q)
                    raw_moves_score_game = np.concatenate((raw_moves_score_game, np.array([r_s])), axis=0)

            if n_game == 0:
                x_episode = x_game
                y_episode = y_game
                raw_moves_score_episode = raw_moves_score_game
            else:
                x_episode = np.concatenate((x_episode, x_game), axis=0)
                y_episode = np.concatenate((y_episode, y_game), axis=0)
                raw_moves_score_episode = np.concatenate((raw_moves_score_episode, raw_moves_score_game), axis=0)

        return x_episode, y_episode, b_episode, q_episode, raw_moves_score_episode

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