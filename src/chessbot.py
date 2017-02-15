import chess
from model import model, WEIGHTS_FILE
import numpy as np

class ChessBot:
	def best_move(self, board, depth=1):
		self.player = board.turn
		max_score = -1
		best_move = None
		for move in board.legal_moves:
			board.push(move)
			score = self.score_move(board, depth)
			board.pop()
			if score > max_score:
				max_score = score
				best_move = move
		return best_move

	def score_move(self, board, alpha=0, beta=1, depth):
		moves = list(board.legal_moves)
		if depth == 0 or len(moves) == 0:
			return self.eval_move(board)
		max_player = board.turn == self.player
		best_score = None
		best_move = None
		for move in moves:
			board.push(move)
			score = self.score_move(board, alpha, beta, depth-1)
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
		return best_score
 
	def eval_move(self, board):
		moves = list(board.legal_moves)
		score = -1
		if len(moves) == 0:
			if board.result() in ['1-0', '0-1']:
				score = 1
			else:
				score = 0.5
		else:
			#eval board by looking at next move
			batch_x = np.zeros(shape=(len(moves), 8, 8, 12), dtype=np.int8)
			for i, move in enumerate(moves):
				board.push(move)
				result = board.result()
				if result in ['1-0', '0-1']:
					score = 0
					board.pop()
					break
				elif result == '1/2-1/2':
					score = 0.5
					board.pop()
					break
				batch_x[i] = self.board_to_matrix(board)
				board.pop()
			if score == -1:
				out = model.predict_proba(batch_x, verbose=0)
				scores = [s[0] for s in out]
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