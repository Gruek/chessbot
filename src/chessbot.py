import chess
from model_conv import model, WEIGHTS_FILE
import numpy as np

class ChessBot:
	def search_moves(self, board, depth=1):
		moves_tree = self.build_tree(board, depth)
		self.score_tree(board, moves_tree)
		best_move = None
		best_score = 0
		for move in moves_tree:
			if move['score'] > best_score:
				best_score = move['score']
				best_move = move['move']
		return best_move

	def score_tree(self, board, moves_tree, depth=0):
		if not moves_tree:
			return self.eval_board(board, depth)
		for move in moves_tree:
			board.push(move['move'])
			move['score'] = self.score_tree(board, move['next_moves'], depth+1)
			board.pop()
		scores = [move['score'] for move in moves_tree]
		max_score = max(scores)
		if depth%2==1:
			return 1-max_score
		return max_score

	def build_tree(self, board, depth, filter=True):
		if depth == 0:
			return None
		moves_tree = []
		for move in board.legal_moves:
			board.push(move)
			if not board.can_claim_threefold_repetition() or not filter:
				next_move = {'move':move, 'next_moves':self.build_tree(board, depth-1), 'score':None}
				moves_tree.append(next_move)
			board.pop()
		if len(moves_tree) == 0:
			return self.build_tree(board, depth, False)
		return moves_tree

	def eval_board(self, board, depth=0):
		moves = list(board.legal_moves)
		if len(moves) == 0:
			if board.result() in ['1-0', '0-1']:
				if depth%2==1:
					return 0
				else:
					return 1
				return 1
			else:
				return 0.3
		#format all the moves into 4d array
		batch_x = np.zeros(shape=(len(moves), 8, 8, 12), dtype=np.int8)
		for i, move in enumerate(moves):
			board.push(move)
			batch_x[i] = self.board_to_matrix(board)
			board.pop()
		#predict
		out = model.predict_proba(batch_x, verbose=0)
		scores = [score[1] for score in out]
		max_score = max(scores)
		if depth%2==1:
			return 1-max_score
		return max_score


	def board_to_matrix(self, board):
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