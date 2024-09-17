from enum import Enum
from typing import Tuple
import copy
import random
import numpy as np

Color = Literal['black', 'white']

class Piece(Enum):
    Pawn: int
    Rook: int
    Knight: int
    Bishop: int
    Queen: int
    King: int


class State:
    def __init__(efficiency: float, configuration: Dict[Tuple[int,int], Tuple(Piece, Color)], turn: Color):
        self.efficiency = efficiency
        self.configuration = configuration
        self.adj: Map[int, List[State]] = []
        self.zobrist_parameters = [ [random.randint(0, 1 << 64 - 1) for _ in range(8)] for _ in range(8)]
        self.zobrist_white = random.randint(0, 1 << 64 - 1)
        self.zobrist_black = random.randint(0, 1 << 64 - 1)

    # random thought, but I wonder if there is a way to use static typing / compilation for chess piece evaluation (i.e. this code?)
    def possible_moves(self, position: Tuple[int,int]):
        piece, cur_color = self.configuration[position]
        
        moves = []
        def linear_moves(x_shift: int, y_shift: int, condition_func):
            nonlocal moves:
            cur_position = (cur_position[0] + shift, cur_position[1] + shift)
            while(condition_func(cur_position) and cur_position not in self.configuration.keys()):
                moves.append(cur_position)
                cur_position = (cur_position[0] + shift, cur_position[1] + shift)

        match piece:
            case Piece.Pawn:
                nxt_pos = [(position[0]+1, position[1]-1), (position[0]+1, position[1]), (position[0]+1, position[1]+1)]
                for i in range(len(nxt_pos)-1, -1, -1):
                    if pos in self.configuration.keys() and self.configuration[pos][1] == cur_color:
                        del nxt_pos[i]
                return nxt_pos
            case Piece.Rook:
                cur_position = position
                linear_moves(-1, 0, lambda x: x[0] >= 0)
                linear_moves(0, -1, lambda x: x[1] >= 0)
                linear_moves(1, 0, lambda x: x[0] < 8)
                linear_moves(0, 1, lambda x: x[1] < 8)
                return moves
            case Piece.Knight:
                nxt_pos = [(position[0]+2, position[1]-1), (position[0]+2, position[1]+1), 
                           (position[0]-2, position[1]+1), (position[0]-2, position[1]-1),
                           (position[0]+1, position[1]-2), (position[0]+1, position[1]+2),
                           (position[0]-1, position[1]-2), (position[0]-1, position[1]+2)]
                for idx in range(len(nxt_pos)-1, -1, -1):
                    if pos in self.configuration.keys() and self.configuration[pos][1] == cur_color:
                        del nxt_pos[idx]
                return nxt_pos
            case Piece.Bishop:
                linear_moves(1, 1, lambda x: x[0] < 8 and x[1] < 8)
                linear_moves(-1, 1, lambda x: x[0] >= 0 and x[1] < 8)
                linear_moves(-1, -1, lambda x: x[0] >= 0 and x[1] >= 0)
                linear_moves(1, -1, lambda x: x[0] < 8 and x[1] >= 0)
                return moves
            case Piece.Queen:
                linear_moves(1, 1, lambda x: x[0] < 8 and x[1] < 8)
                linear_moves(-1, 1, lambda x: x[0] >= 0 and x[1] < 8)
                linear_moves(-1, -1, lambda x: x[0] >= 0 and x[1] >= 0)
                linear_moves(1, -1, lambda x: x[0] < 8 and x[1] >= 0)
                linear_moves(-1, 0, lambda x: x[0] >= 0)
                linear_moves(0, -1, lambda x: x[1] >= 0)
                linear_moves(1, 0, lambda x: x[0] < 8)
                linear_moves(0, 1, lambda x: x[1] < 8)
                return moves
            case Piece.King:
                dx = [-1, 0, 1]
                dy = [-1, 0, 1]
                nxt_pos = []
                for x in dx:
                    for y in dy:
                        if (x == 0 and y == 0):
                            continue
                        if (0 <= cur_position[0] + x < 8 and 0 <= cur_position[1] + y < 8):
                            nxt_pos.append( (cur_position[0] + x, cur_position[1] + y))
                for idx in range(len(nxt_pos)-1, -1, -1):
                    if pos in self.configuration.keys() and self.configuration[pos][1] == cur_color:
                        del nxt_pos[idx]
                return nxt_pos   

    @static
    def zobrist_hashing(board: List[List[Piece]], turn: Literal['black', 'white']):
        h = 0
        if turn == 'black':
            h = h ^ self.zobrist_black
        else:
            h = h ^ self.zobrist_black

        for (i, line) in enumerate(board):
            for (j, piece) in enumerate(line):
                if piece != Piece.Empty:
                    h = h ^ self.zobrist_parameters[i][j]
        return h
    
    
    def random_move(self):
        random_piece = random.sample(list(configuration.keys()), 1)
        

    def gen_state(move: Tuple[Tuple[int,int], Tuple[int,int]]) -> Optional[State]:
        cur_piece = self.configuration[move[0][0]][move[0][1]]
        configuration: Dict[Tuple[int,int], Piece] = copy.deepcopy(self.configuration)
        if not evaluate_move(move, cur_piece):
            return None

        configuration[move[1]] = cur_piece

        next_turn = 'white' if self.turn == 'black' else 'black' 
        nboard_hash = State.zobrist_hashing(configuration, next_turn) 
        if nboard_hash not in self.adj.keys():
            self.adj[nboard_hash] = State(0.0, configuration, next_turn)
        return self.adj[nboard_hash]



def mcts(root_state: Optional[State]):
    


if __name__ == '__main__':

