import chess
import chess.pgn
from model import model, WEIGHTS_FILE
import numpy as np
from sunfish import sunfish
from chessbot import ChessBot

chessbot = ChessBot()

class Trainer:
	def play_vs_self(self):
		board = chess.Board()

		# simulate a game vs self
		while not board.is_game_over():
			move = self.best_move(board, filter=True)
			#move = chessbot.best_move(board, depth=1)
			board.push(move)

		draw = board.result() == '1/2-1/2'
		moves = len(board.move_stack)

		self.train_from_match(board)
		return draw, moves

	def train_vs_self(self):
		iterations = 0
		while True:
			iterations += 1
			games = 0
			draws = 0
			total_moves = 0
			for i in range(50):
				draw, moves = self.play_vs_self()
				draws += draw
				total_moves += moves
				games += 1
			model.save_weights(WEIGHTS_FILE)
			print('Draw rate:', draws / games, 'Avg moves:', total_moves / games)
			if iterations%20==0:
				print('Win Rate:', self.test_winrate())

	def best_move(self, board, eval=False, filter=False):
		# filter out shit moves
		decent_moves = []
		if filter:
			for move in board.legal_moves:
				board.push(move)
				if board.result() in ['1-0', '0-1']:
					board.pop()
					return move
				if not board.can_claim_threefold_repetition():
					decent_moves.append(move)
				board.pop()
			if len(decent_moves) == 0:
				decent_moves = board.legal_moves
		else:
			decent_moves = board.legal_moves

		#format all the moves into 4d array
		batch_x = np.zeros(shape=(len(decent_moves), 8, 8, 12), dtype=np.int8)
		moves = []
		for i, move in enumerate(decent_moves):
			board.push(move)
			state = self.board_to_matrix(board)
			batch_x[i] = state
			board.pop()
			moves.append(move)

		out = model.predict_proba(batch_x, verbose=0)

		best_score = 0
		best_move = None
		#return best move
		for i, score in enumerate(out):
			if score[0] > best_score:
				best_score = score[0]
				best_move = moves[i]
			if eval:
				print(moves[i], score[0])
		return best_move

	def board_to_matrix(self, board, pov=None):
		if not pov:
			#assume move has been made
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

	def train_vs_sunfish(self):
		while True:
			games = 0
			wins = 0
			for i in range(50):
				win = self.play_vs_sunfish()
				if win:
					wins += 1
				games += 1
			print('Win rate:', wins/games)
			model.save_weights(WEIGHTS_FILE)

	def play_vs_sunfish(self, eval=False):
		board = chess.Board()
		sunfish_board = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
		sunfish_searcher = sunfish.Searcher()
		sun_color = np.random.randint(2)

		while not board.is_game_over():
			if board.turn == sun_color:
				sun_move, score = sunfish_searcher.search(sunfish_board, 0.05)
				if board.turn == chess.BLACK:
					move_str = sunfish.render(119-sun_move[0]) + sunfish.render(119 - sun_move[1])
				else:
					move_str = sunfish.render(sun_move[0]) + sunfish.render(sun_move[1])
				move = chess.Move.from_uci(move_str)
				if int(move.to_square/8) in [0, 7]:
					if board.piece_at(move.from_square).piece_type == chess.PAWN:
						move.promotion = chess.QUEEN
			else:
				#move = self.best_move(board)
				move = chessbot.best_move(board)

			move_str = str(move)
			sun_move = sunfish.parse(move_str[0:2]), sunfish.parse(move_str[2:4])
			if board.turn == chess.BLACK:
				move_str = sunfish.render(119-sun_move[0]) + sunfish.render(119 - sun_move[1])
				sun_move = sunfish.parse(move_str[0:2]), sunfish.parse(move_str[2:4])

			board.push(move)

			sunfish_board = sunfish_board.move(sun_move)
			sunfish_board.rotate()

		result = board.result()
		# draw
		winner = 2
		if result == '1-0':
			#white
			winner = chess.WHITE
		elif result == '0-1':
			#black
			winner = chess.BLACK
		won = False
		if winner < 2 and winner != sun_color:
			won = True
		if eval:
			return board, won
		self.train_from_match(board, result)
		return won

	def test_winrate(self):
		wins = 0
		games = 0
		for i in range(4):
			games += 1
			board, won = self.play_vs_sunfish(eval=True)
			wins += won
		return wins/games

	def train_from_pros(self):
		files = [
			"ficsgamesdb_2016_standard2000_nomovetimes_1435145.pgn",
			"ficsgamesdb_2015_standard2000_nomovetimes_1441190.pgn",
			"ficsgamesdb_2014_standard2000_nomovetimes_1441191.pgn"
		]
		games = 0
		epoch = 0
		file_idx = np.random.randint(len(files))
		file = open(files[epoch])
		while True:
			game = chess.pgn.read_game(file)
			if not game:
				epoch += 1
				file_idx += 1
				file = open(files[file_idx%len(files)])
				print('Win Rate:', self.test_winrate())
				continue
			board = game.end().board()
			if len(board.move_stack) < 2:
				continue

			batch_x = np.zeros(shape=(len(board.move_stack)*2, 8, 8, 12), dtype=np.int8)
			batch_y = np.zeros(shape=(len(board.move_stack)*2, 2), dtype=np.float)

			#if board.result() not in ['1-0', '0-1']:
			#	continue

			for i in range(len(board.move_stack)):
				# pro moves == good
				batch_x[2*i] = self.board_to_matrix(board)
				batch_y[2*i] = [1, 0]
				board.pop()
				# random move == bad
				possible_moves = list(board.legal_moves)
				random_move = possible_moves[np.random.randint(len(possible_moves))]
				board.push(random_move)
				batch_x[2*i+1] = self.board_to_matrix(board)
				batch_y[2*i+1] = [0, 1]
				board.pop()
			model.train_on_batch(batch_x, batch_y)
			games += 1

			if games % 1000 == 0:
				model.save_weights(WEIGHTS_FILE)
				print('Games:', games, 'Epoch:', epoch)

	def train_from_match(self, board, result=None):
		if not result:
			result = board.result()
		# draw
		winner = 2
		if result == '1-0':
			#white
			winner = chess.WHITE
		elif result == '0-1':
			#black
			winner = chess.BLACK

		moves = list(board.move_stack)
		moves_num = len(moves)
		while moves_num > 0:
			batch_size = max(moves_num, 50)
			moves_num -= batch_size
			batch_x = np.zeros(shape=(batch_size, 8, 8, 12), dtype=np.int8)
			batch_y = np.zeros(shape=(batch_size, 2), dtype=np.float)

			for i in range(batch_size):
				score = 1 #0.5 + ((len(moves)-i)/len(moves))/2
				if winner == 2: #DRAW
					score = 0.5
				last_turn = not board.turn
				batch_x[i] = self.board_to_matrix(board)
				batch_y[i] = [score, 1-score] if winner == last_turn else [1-score, score]
				board.pop()
			model.train_on_batch(batch_x, batch_y)
