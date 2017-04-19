import chess
from model_twostate import model, WEIGHTS_FILE
import numpy as np
import time

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class ChessBot:
    def __init__(self):
        self.max_cache = 50
        self.cache = [{}] * self.max_cache

    def clear_cache(self):
        self.cache = [{}] * self.max_cache

    def best_move(self, board, depth=10, eval=False, think_time=30):
        #clean old cache
        self.cache.insert(0, {})
        self.cache = self.cache[:self.max_cache]

        self.player = board.turn
        self.time_limit = time.time() + think_time
        score, move, dead = self.score_move(board, depth)
        if eval:
            print(score)
        return move

    def score_move(self, board, depth, alpha=0, beta=1):
        moves = list(board.legal_moves)
        if len(moves) == 0 or depth == 0:
            move_score = self.eval_move(board)
            return move_score['score'], None, True
        max_player = board.turn == self.player
        best_score = None
        best_move = None

        move_scores = self.possible_moves(moves, board)

        while True:
            move_scores.sort(key=lambda x: x['score'], reverse=max_player)
            best_move = move_scores[0]['move']
            best_score = move_scores[0]['score']
            if alpha >= best_score or best_score >= beta:
                break
            if time.time() > self.time_limit:
                break

            move_to_eval = None
            i = 0
            while True:
                if i >= len(move_scores):
                    break
                if not move_scores[i]['dead']:
                    move_to_eval = move_scores[i]
                    break
                i += 1
            if not move_to_eval:
                return best_score, best_move, True
            temp_alpha = alpha
            temp_beta = beta
            if len(move_scores) > i+1:
                temp_limit = move_scores[i+1]['score']
                if max_player:
                    temp_alpha = max([temp_alpha, temp_limit])
                else:
                    temp_beta = min([temp_beta, temp_limit])
            #go deeper
            board.push(move_to_eval['move'])
            score, bm, dead = self.score_move(board, depth-1, temp_alpha, temp_beta)
            #cache updated score
            board_hash = board.board_fen() + str(int(board.turn))
            self.cache[0][board_hash] = {'score': score, 'dead': dead}
            move_to_eval['dead'] = dead
            move_to_eval['score'] = score
            board.pop()
            
        return best_score, best_move, False
 
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
                # dead = cached['dead']
            else:
                # run neural network
                batch_x = np.zeros(shape=(1, 2, 8, 8, 12), dtype=np.int8)
                temp_move = board.pop()
                pov = board.turn
                batch_x[0][0] = self.board_to_matrix(board, pov)
                board.push(temp_move)
                batch_x[0][1] = self.board_to_matrix(board, pov)
                out = model.predict(batch_x, verbose=0)
                score = out[0][0]
                #cache score
                self.cache[0][board_hash] = {'score': score, 'dead': dead }
            if score > 0.5 and board.can_claim_threefold_repetition():
                score = 0.5
                dead = True
        if not board.turn == self.player:
            return {'score': score, 'dead': dead}
        else:
            return {'score': 1-score, 'dead': dead}

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
