import chess
import chess.pgn
import numpy as np
from sunfish import sunfish
from chessbot import ChessBot, model, WEIGHTS_FILE
from pystockfish import Engine
from datetime import datetime

chessbot = ChessBot()
stockfish = Engine(depth=20, param={"Threads": 6, "Hash": 12288})
shitfish = Engine(depth=0, param={"Threads": 6, "Hash": 12288})

class Trainer:
    def play_vs_self(self):
        board = chess.Board()

        # simulate a game vs self
        while not board.is_game_over():
            move = chessbot.best_move(board, 0)
            board.push(move)

        result = board.result()
        draw = result == '1/2-1/2'
        moves = len(board.move_stack)

        self.train_from_match(board, result, use_stockfish=True)
        return draw, moves

    def train_vs_self(self):
        iterations = 0
        while True:
            iterations += 1
            games = 0
            draws = 0
            total_moves = 0
            for i in range(10):
                draw, moves = self.play_vs_self()
                draws += draw
                total_moves += moves
                games += 1
            model.save_weights(WEIGHTS_FILE, overwrite=True)
            print('Draw rate:', draws / games, 'Avg moves:', total_moves / games)
            print(self.validation())
            #if iterations%20==0:
                #print('Win Rate:', self.test_winrate())

    def best_move(self, board, eval=False):
        return chessbot.best_move(board, 0, eval)

    def train_vs_stockfish(self):
        while True:
            games = 0
            wins = 0
            for i in range(100):
                win = self.play_vs_stockfish(fish=shitfish, depth=0)
                if win:
                    wins += 1
                games += 1
            model.save_weights(WEIGHTS_FILE, overwrite=True)
            print(datetime.now(), 'Win rate:', wins/games, self.validation())

    def play_vs_sunfish(self, eval=False):
        board = chess.Board()
        sunfish_board = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        sunfish_searcher = sunfish.Searcher()
        sun_color = np.random.randint(2)

        while not board.is_game_over():
            if board.turn == sun_color:
                sun_move, score = sunfish_searcher.search(sunfish_board, 5)
                if board.turn == chess.BLACK:
                    move_str = sunfish.render(119-sun_move[0]) + sunfish.render(119 - sun_move[1])
                else:
                    move_str = sunfish.render(sun_move[0]) + sunfish.render(sun_move[1])
                move = chess.Move.from_uci(move_str)
                if int(move.to_square/8) in [0, 7]:
                    if board.piece_at(move.from_square).piece_type == chess.PAWN:
                        move.promotion = chess.QUEEN
            else:
                move = self.best_move(board)
                # move = chessbot.best_move(board)

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
        print(result, won, len(board.move_stack))
        if eval:
            return board, won
        self.train_from_match(board, result)
        return won

    def play_vs_stockfish(self, eval=False, fish=stockfish, depth=4):
        board = chess.Board()
        fish.newgame()
        stockfish_color = np.random.randint(2)

        while not board.is_game_over():
            if board.turn == stockfish_color:
                fish.setfenposition(board.fen())
                move_str = fish.bestmove()['move']
            else:
                move = chessbot.best_move(board, depth=depth)
                move_str = str(move)

            try:
                move = chess.Move.from_uci(move_str)
                board.push(move)
            except:
                print(move_str)
                return board, False

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
        if winner < 2 and winner != stockfish_color:
            won = True
        # print(result, won, len(board.move_stack))
        if eval:
            return board, won
        self.train_from_match(board, result)
        return won

    def test_winrate(self):
        wins = 0
        games = 0
        for i in range(4):
            games += 1
            board, won = self.play_vs_stockfish(True, shitfish)
            wins += won
        return wins/games

    def train_from_pros(self):
        files = [
            # "ficsgamesdb_2016_standard2000_nomovetimes_1435145.pgn",
            # "ficsgamesdb_2015_standard2000_nomovetimes_1441190.pgn",
            # "ficsgamesdb_2014_standard2000_nomovetimes_1441191.pgn",
            "ficsgamesdb_2016_chess_nomovetimes_1445486.pgn"
        ]
        games = 0
        epoch = 0
        file_idx = np.random.randint(len(files))
        file = open(files[file_idx])
        for i in range(945000):
            game = chess.pgn.read_game(file)
            games += 1

        while True:
            t1 = datetime.now()
            try:
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
            except:
                continue

            if not game.headers["Result"] in ['1-0', '0-1']:
                continue

            self.train_from_match(board, game.headers["Result"])

            # moves_num = len(board.move_stack)
            # while moves_num > 0:
            # 	batch_size = min([moves_num, 25])
            # 	moves_num -= batch_size

            # 	batch_x = np.zeros(shape=(batch_size*2, 8, 8, 12), dtype=np.int8)
            # 	batch_y = np.zeros(shape=(batch_size*2, 2), dtype=np.float)

            # 	for i in range(batch_size):
            # 		# pro moves == good
            # 		batch_x[2*i] = self.board_to_matrix(board)
            # 		batch_y[2*i] = [1, 0]
            # 		board.pop()
            # 		# random move == bad
            # 		possible_moves = list(board.legal_moves)
            # 		random_move = possible_moves[np.random.randint(len(possible_moves))]
            # 		board.push(random_move)
            # 		batch_x[2*i+1] = self.board_to_matrix(board)
            # 		batch_y[2*i+1] = [0, 1]
            # 		board.pop()
            # 	model.train_on_batch(batch_x, batch_y)

            games += 1
            if games % 5000 == 0:
                model.save_weights(WEIGHTS_FILE, overwrite=True)
                print('Games:', games, 'Epoch:', epoch)
                print(self.validation())
                t2 = datetime.now()
                print(t2- t1)
                t1 = t2

    def train_from_match(self, board, result=None, use_stockfish=False):
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
            batch_size = min([moves_num, 50])
            moves_num -= batch_size
            train_size = batch_size*2 if use_stockfish else batch_size
            batch_x = np.zeros(shape=(train_size, 2, 8, 8, 12), dtype=np.int8)
            batch_y = np.zeros(shape=(train_size, 2), dtype=np.float)

            for i in range(batch_size):
                score = 1 #0.5 + ((len(moves)-i)/len(moves))/2
                if winner == 2: #DRAW
                    score = 0.5
                pov = not board.turn
                if use_stockfish:
                    #move made
                    batch_x[i*2][1] = chessbot.board_to_matrix(board, pov)
                    board.pop()
                    batch_x[i*2][0] = chessbot.board_to_matrix(board, pov)
                    batch_y[i*2] = [1-score, score]
                    #stockfish move
                    stockfish.setfenposition(board.fen())
                    move_str = stockfish.bestmove()['move']
                    move = chess.Move.from_uci(move_str)
                    batch_x[i*2+1][0] = batch_x[i*2][0]
                    board.push(move)
                    batch_x[i*2+1][1] = chessbot.board_to_matrix(board, pov)
                    board.pop()
                    batch_y[i*2+1] = [score, 1-score]
                else:
                    batch_x[i][1] = chessbot.board_to_matrix(board, pov)
                    board.pop()
                    batch_x[i][0] = chessbot.board_to_matrix(board, pov)
                    batch_y[i] = [score, 1-score] if winner == pov else [1-score, score]
            model.train_on_batch(batch_x, batch_y)
        chessbot.clear_cache()

    def validation(self):
        file = "ficsgamesdb_2016_standard2000_nomovetimes_1435145.pgn"
        file = open(file)
        sample = 0
        correct = 0
        max_sample = 1000

        while True:
            if sample > max_sample:
                return correct/sample
            try:
                game = chess.pgn.read_game(file)
                if not game:
                    return correct/sample
                board = game.end().board()
                if len(board.move_stack) <= 2:
                    continue
            except:
                continue

            result = game.headers["Result"]
            if not result in ['1-0', '0-1']:
                continue

            #eval the second half of the game
            moves_num = len(board.move_stack)//2
            moves_num = min([5, moves_num])

            batch_x = np.zeros(shape=(moves_num, 2, 8, 8, 12), dtype=np.int8)
            move_turn = not board.turn

            for i in range(moves_num):
                pov = not board.turn
                batch_x[i][1] = chessbot.board_to_matrix(board, pov)
                board.pop()
                batch_x[i][0] = chessbot.board_to_matrix(board, pov)
            out = model.predict(batch_x, verbose=0)
            winner = result == '1-0'
            for score in out:
                if winner == move_turn and score[0] > 0.5:
                    correct += 1
                if winner != move_turn and score[0] < 0.5:
                    correct += 1
                sample += 1
                move_turn = not move_turn

    def get_match_from_data(self, dataset):
        game = chess.pgn.read_game(dataset['file'])
        if not game:
            dataset['file_index'] += 1
            dataset['file_index'] %= len(dataset['files'])
            dataset['file'] = open(dataset['files'][dataset['file_index']])
            return self.get_match_from_data(dataset)
        return game

    def train_from_data(self):
        pro_data = {
            'files': [
                "ficsgamesdb_2016_standard2000_nomovetimes_1435145.pgn",
                "ficsgamesdb_2015_standard2000_nomovetimes_1441190.pgn",
                "ficsgamesdb_2014_standard2000_nomovetimes_1441191.pgn",
            ],
            'file_index': 0,
            'file': None
        }
        standard_data = {
            'files': ["ficsgamesdb_2016_chess_nomovetimes_1445486.pgn"],
            'file_index': 0,
            'file': None
        }
        pro_data['file_index'] = np.random.randint(len(pro_data['files']))
        pro_data['file'] = open(pro_data['files'][pro_data['file_index']])
        standard_data['file_index'] = np.random.randint(len(standard_data['files']))
        standard_data['file'] = open(standard_data['files'][standard_data['file_index']])

        games = 0
        pro_game = True
        skips = 0

        while True:
            t1 = datetime.now()
            try:
                if pro_game:
                    game = self.get_match_from_data(pro_data)
                else:
                    game = self.get_match_from_data(standard_data)
                board = game.end().board()
                if len(board.move_stack) < 2:
                    continue
            except:
                continue
            
            result = game.headers["Result"]
            if not result in ['1-0', '0-1']:
                continue
            if skips > 0:
                games += 1
                skips -= 1
                continue

            if not pro_game:
                self.train_from_match(board, result)
            else:
                winner = 2
                if result == '1-0':
                    winner = 1
                elif result == '0-1':
                    winner = 0
                moves_num = len(board.move_stack)
                while moves_num > 0:
                    batch_size = min([moves_num, 25])
                    moves_num -= batch_size

                    batch_x = np.zeros(shape=(batch_size*2, 2, 8, 8, 12), dtype=np.int8)
                    batch_y = np.zeros(shape=(batch_size*2, 2), dtype=np.float)

                    for i in range(batch_size):
                        pov = not board.turn
                        # pro move == good
                        batch_x[2*i][1] = chessbot.board_to_matrix(board, pov)
                        board.pop()
                        batch_x[2*i][0] = chessbot.board_to_matrix(board, pov)
                        batch_y[2*i] = [1, 0]
                        # random move == bad
                        possible_moves = list(board.legal_moves)
                        random_move = possible_moves[np.random.randint(len(possible_moves))]
                        batch_x[2*i+1][0] = batch_x[2*i][0]
                        board.push(random_move)
                        batch_x[2*i+1][1] = chessbot.board_to_matrix(board, pov)
                        batch_y[2*i+1] = [0, 1]
                        board.pop()
                    model.train_on_batch(batch_x, batch_y)

            games += 1
            pro_game = not pro_game
            if games % 5000 == 0:
                model.save_weights(WEIGHTS_FILE, overwrite=True)
                print('Games:', games)
                print(self.validation())
                t2 = datetime.now()
                print(t2- t1)
                t1 = t2
