import os
import chess
from model_conv import get_model, WEIGHTS_FILE, FILE_EXT
import numpy as np
import time
import shutil

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class ChessBot:
    def __init__(self):
        self.SCORE_MARGIN = 0.005
        self.max_cache = 100
        self.cache = [{}] * self.max_cache
        self.model = get_model()
        self.model_name = None

    def clear_cache(self):
        self.cache = [{}] * self.max_cache

    def best_move(self, board, depth=6, eval=False, think_time=30):
        self.prepare_cache()
        self.states_evaled = 0

        self.load_model(self.choose_model(board))
        self.player = board.turn
        self.time_limit = time.time() + think_time
        score_move = self.score_move(board, depth)
        if eval:
            print(score_move['score'], self.states_evaled)
        return score_move['move']

    def prepare_cache(self):
        #clear dead states
        for m in self.cache[0].values():
            m['dead'] = False
        #clean old cache
        self.cache.insert(0, {})
        self.cache = self.cache[:self.max_cache]

    def score_move(self, board, depth, alpha=0, beta=1):
        moves = list(board.legal_moves)
        if len(moves) == 0 or depth == 0:
            move_score = self.eval_move(board)
            move_score['dead'] = True
            return move_score
        max_player = board.turn == self.player
        best_move = None

        move_scores = self.possible_moves(moves, board)
        move_scores.sort(key=lambda x: x['score'], reverse=max_player)
        moves_temp_order = list(move_scores)

        while True:
            best_move = move_scores[0]
            if time.time() > self.time_limit:
                break
            move_to_eval = None
            i = 0
            while True:
                if i >= len(moves_temp_order):
                    break
                if not moves_temp_order[i]['dead']:
                    move_to_eval = moves_temp_order[i]
                    break
                i += 1
            if not move_to_eval:
                best_move['dead'] = True
                # print(move_to_eval)
                return best_move
            #if move to eval is within limits
            score = move_to_eval['score']
            if alpha >= score or score >= beta:
                best_move['temp_score'] = move_to_eval['score']
                best_move['dead'] = False
                # print(move_to_eval)
                return best_move
            # print(move_to_eval)
            temp_alpha = alpha
            temp_beta = beta
            if len(move_scores) > i+1:
                temp_limit = move_scores[i+1]['score']
                if max_player:
                    temp_alpha = max([temp_alpha, temp_limit - self.SCORE_MARGIN])
                else:
                    temp_beta = min([temp_beta, temp_limit + self.SCORE_MARGIN])
            #go deeper
            board.push(move_to_eval['move'])
            move_to_eval_temp = self.score_move(board, depth-1, temp_alpha, temp_beta)
            #cache updated score
            board_hash = board.board_fen() + str(int(board.turn))
            self.cache[0][board_hash] = {
                'score': move_to_eval_temp['score'],
                'dead': move_to_eval_temp['dead'],
                'train_fen': board.fen(),
                'train_model': self.model_name
            }
            move_to_eval['dead'] = move_to_eval_temp['dead']
            move_to_eval['score'] = move_to_eval_temp['score']
            move_to_eval['temp_score'] = move_to_eval_temp['temp_score']
            move_scores.remove(move_to_eval)
            moves_temp_order.remove(move_to_eval)
            self.insert_sorted(move_scores, move_to_eval, lambda x: x['score'], max_player)
            self.insert_sorted(moves_temp_order, move_to_eval, lambda x: x['temp_score'], max_player)
            board.pop()
        
        return best_move
 
    def eval_move(self, board):
        moves = list(board.legal_moves)
        score = -1
        dead = True
        result = board.result()
        if result in ['1-0', '0-1']:
            score = 1
        elif result == '1/2-1/2':
            score = 0.5
        else:
            dead = False
            board_hash = board.board_fen() + str(int(board.turn))
            cached = self.from_cache(board_hash)
            if cached:
                score = cached['score']
                dead = cached['dead']
            else:
                # run neural network
                # print(len(board.move_stack))
                self.states_evaled += 1
                batch_x = np.zeros(shape=(1, 8, 8, 12), dtype=np.int8)
                batch_x[0] = self.board_to_matrix(board)
                out = self.model.predict(batch_x, verbose=0)
                score = out[0][0]
                #cache score
                self.cache[0][board_hash] = {'score': score, 'dead': dead}
            if score > 0.5 and board.can_claim_threefold_repetition():
                score = 0.5
                dead = True
        move_score = {'dead': dead, 'move': None, 'score' : score if board.turn != self.player else 1 - score}
        move_score['temp_score'] = move_score['score']
        return move_score

    def possible_moves(self, moves, board):
        move_scores = []
        for move in moves:
            board.push(move)
            move_score = self.eval_move(board)
            move_score['move'] = move
            move_scores.append(move_score)
            board.pop()
        return move_scores

    def board_to_matrix(self, board, pov=None):
        if not pov:
            pov = not board.turn
        state = np.zeros(shape=(8, 8, 12), dtype=np.int8)
        for x in range(8):
            for y in range(8):
                piece = board.piece_at(x*8+y)
                if piece:
                    enemy = piece.color != pov
                    piece_idx = enemy * 6 + piece.piece_type - 1
                    #rotate the board for black
                    if pov == chess.BLACK:
                        y = 7-y
                        x = 7-x
                    state[x][y][piece_idx] = 1
        return state

    def from_cache(self, board_hash):
        for cachen in self.cache:
            if board_hash in cachen:
                ret = cachen[board_hash]
                del cachen[board_hash]
                #move to primary cache
                self.cache[0][board_hash] = ret
                return ret
        return None

    def insert_sorted(self, l, item, key=lambda x: x, reverse=False):
        i = 0
        val = key(item)
        while i < len(l):
            index_val = key(l[i])
            if reverse:
                if index_val < val:
                    break
            else:
                if index_val > val:
                    break
            i += 1
        l.insert(i, item)

    def count_pieces(self, board):
        pieces = 0
        for x in range(8):
            for y in range(8):
                piece = board.piece_at(x*8+y)
                if piece:
                    pieces += 1
        return pieces

    def choose_model(self, board):
        if len(board.move_stack) < 12:
            return 'early'
        if self.count_pieces(board) < 12:
            return 'late'
        return 'mid'

    def get_model_file(self, name, temp=False):
        fn = WEIGHTS_FILE + '_' + name
        if temp:
            fn += '_temp'
        fn += FILE_EXT
        return fn

    def load_model(self, name):
        if name != self.model_name:
            self.model_name = name
            filename = self.get_model_file(self.model_name)
            if os.path.isfile(filename):
                # print('loading', self.model_name)
                self.model.load_weights(filename)

    def save_model(self):
        temp_file = self.get_model_file(self.model_name, True)
        real_file = self.get_model_file(self.model_name)
        try:
            self.model.save_weights(temp_file)
            os.remove(real_file)
            shutil.move(temp_file, real_file)
        except KeyboardInterrupt:
            os.remove(temp_file)
            self.save_model()
            exit()

