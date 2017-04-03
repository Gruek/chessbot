import chess
from model_twostate import model, WEIGHTS_FILE
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class ChessBot:
    def __init__(self):
        self.max_cache = 50
        self.cache = [{}] * self.max_cache

    def clear_cache(self):
        self.cache = [{}] * self.max_cache

    def best_move(self, board, depth=4, eval=False):
        #clean old cache
        self.cache.insert(0, {})
        self.cache = self.cache[:self.max_cache]

        self.player = board.turn
        score, move = self.score_move(board, depth)
        if eval:
            print(score)
        return move

    def score_move(self, board, depth, alpha=0, beta=1, breadth_range=0.8):
        moves = list(board.legal_moves)
        if depth == 0 or len(moves) == 0:
            score, move = self.eval_move(board)
            #if perfect score then sort by depth
            if score == 1:
                score += depth
            elif score == 0:
                score -= depth
            return score, move
        max_player = board.turn == self.player
        best_score = None
        best_move = None
        #get top max_breadth moves
        move_scores = self.score_moves(moves, board)
        move_scores.sort(key=lambda x: x['score'], reverse=True)
        scores = [score['score'] for score in move_scores]
        softmaxed_scores = softmax(scores)
        breadth = 0
        moves = []
        for i, move in enumerate(move_scores):
            if breadth > breadth_range:
                break
            breadth += softmaxed_scores[i]
            moves.append(move['move'])

        #go deeper
        for move in moves:
            board.push(move)
            score, bm = self.score_move(board, depth-1, alpha, beta)
            board.pop()
            if best_score == None:
                best_score = score
                best_move = move
            if max_player: #MAX
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max([alpha, score])
            else: #MIN
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min([beta, score])
            if beta <= alpha:
                #prune
                break
        return best_score, best_move
 
    def eval_move(self, board):
        moves = list(board.legal_moves)
        score = -1
        move = None
        result = board.result()
        if result in ['1-0', '0-1']:
            score = 1
        elif result == '1/2-1/2':
            score = 0.5
        else:
            board_hash = board.board_fen() + str(int(board.turn))
            cached = self.from_cache(board_hash)
            if cached:
                score = cached
            else:
                #eval board by looking at next move
                scores = self.score_moves(moves, board)
                for move_score in scores:
                    if move_score['score'] > score:
                        score = move_score['score']
                        move = move_score['move']
                score = 1 - score
                if score > 0.5 and board.can_claim_threefold_repetition():
                    score = 0.5
                #cache score
                self.cache[0][board_hash] = score
        if not board.turn == self.player:
            return score, move
        else:
            return 1-score, move

    def score_moves(self, moves, board):
        scores = []
        if len(moves) > 0:
            batch_x = np.zeros(shape=(len(moves), 2, 8, 8, 12), dtype=np.int8)
            pov = board.turn
            for i, move in enumerate(moves):
                batch_x[i][0] = self.board_to_matrix(board, pov)
                board.push(move)
                if board.result() in ['1-0', '0-1']:
                    board.pop()
                    return [{'score': 1, 'move': move}]
                batch_x[i][1] = self.board_to_matrix(board, pov)
                board.pop()
            out = model.predict(batch_x, verbose=0)
            for i, score in enumerate(out):
                scores.append({'score': score[0], 'move': moves[i]})
        return scores

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
                #move to primary cache
                self.cache[0][board_hash] = ret
                return ret
        return None
