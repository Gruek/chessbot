import chess
from model_conv import model, WEIGHTS_FILE
import numpy as np

class ChessBot:
	def best_move(self, board, depth=1):
		self.player = board.turn
		max_score = 0
		best_move = None
		for move in board.legal_moves:
			board.push(move)
			score = self.score_move(board, depth)
			board.pop()
			if score > max_score:
				max_score = score
				best_move = move
		return best_move

	def score_move(self, board, depth):
		moves = list(board.legal_moves)
		if depth == 0 or len(moves) == 0:
			return self.eval_move(board)
		max_score = 0
		min_score = 1
		for move in moves:
			board.push(move)
			score = self.score_move(board, depth-1)
			board.pop()
			if score > max_score:
				max_score = score
			if score < min_score:
				min_score = score
		if board.turn == self.player:
			return max_score
		return min_score

	def eval_move(self, board):
		moves = list(board.legal_moves)
		score = 0
		if len(moves) == 0:
			if board.result() in ['1-0', '0-1']:
				score = 1
			else:
				score = 0.3
		else:
			#eval board by looking at next move
			batch_x = np.zeros(shape=(len(moves), 8, 8, 12), dtype=np.int8)
			for i, move in enumerate(moves):
				board.push(move)
				batch_x[i] = self.board_to_matrix(board)
				board.pop()
			out = model.predict_proba(batch_x, verbose=0)
			scores = [s[1] for s in out]
			max_score = max(scores)
			score = 1 - max_score
		if not board.turn == self.player:
			return score
		else:
			return 1-score

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