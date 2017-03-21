import chess
from model import model, WEIGHTS_FILE
import numpy as np

class ChessBot:
	def __init__(self):
		self.max_cache = 2
		self.cache = [{}] * self.max_cache

	def best_move(self, board, depth=5, eval=False):
		#clean old cache
		self.cache.insert(0, {})
		self.cache = self.cache[:self.max_cache]

		self.player = board.turn
		score, move = self.score_move(board, depth)
		if eval:
			print(score)
		return move

	def score_move(self, board, depth, alpha=0, beta=1, max_breadth=5):
		moves = list(board.legal_moves)
		if depth == 0 or len(moves) == 0:
			score = self.eval_move(board)
			#if perfect score then sort by depth
			if score == 1:
				score += depth
			elif score == 0:
				score -= depth
			return score, None
		max_player = board.turn == self.player
		best_score = None
		best_move = None
		#get top 5 moves
		scores = self.score_moves(moves, board)
		scores.sort(key = lambda x: x['score'], reverse=True)
		moves = [score['move'] for score in scores[:max_breadth]]

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
				batch_x = np.zeros(shape=(len(moves), 8, 8, 12), dtype=np.int8)
				for i, move in enumerate(moves):
					board.push(move)
					result = board.result()
					if result in ['1-0', '0-1']:
						score = 0
						board.pop()
						break
					batch_x[i] = self.board_to_matrix(board)
					board.pop()
				if score == -1:
					out = model.predict(batch_x, verbose=0)
					scores = [s[0] for s in out]
					max_score = max(scores)
					score = 1 - max_score
				if score > 0.5 and board.can_claim_threefold_repetition():
					score = 0.5
				#cache score
				self.cache[0][board_hash] = score
		if not board.turn == self.player:
			return score
		else:
			return 1-score

	def score_moves(self, moves, board):
		scores = []
		for move in moves:
			board.push(move)
			if board.result() in ['1-0', '0-1']:
				board.pop()
				return [{'score': 1, 'move': move}]
			board.pop()
		if len(moves) > 0:
			batch_x = np.zeros(shape=(len(moves), 8, 8, 12), dtype=np.int8)
			for i, move in enumerate(moves):
				board.push(move)
				batch_x[i] = self.board_to_matrix(board)
				board.pop()
			out = model.predict(batch_x, verbose=0)
			for i, score in enumerate(out):
				scores.append({'score': score[0], 'move': moves[i]})
		return scores

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

	def from_cache(self, board_hash):
		for cachen in self.cache:
			if board_hash in cachen:
				ret = cachen[board_hash]
				#move to primary cache
				self.cache[0][board_hash] = ret
				return ret
		return None