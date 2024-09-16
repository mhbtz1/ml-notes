from pydantic import BaseModel
from typing import Tuple, 
import copy

class Piece(BaseModel):
    Pawn: int
    Rook: int
    Knight: int
    Bishop: int
    Queen: int
    King: int



class State:
    def __init__(prob: float, configuration: List[List[Piece]], turn: Literal['black', 'white']):
        self.prob = prob
        self.configuration = configuration
        self.adj: Map[int, List[State]] = []

    def evaluate_move(move: Tuple[Tuple[int,int], Tuple[int,int]], piece: Piece):
        pass
    
    @static
    def zobrist_hashing(board: List[List[Piece]], turn: Literal['black', 'white']):
        if turn == 'black':

    def gen_state(move: Tuple[Tuple[int,int], Tuple[int,int]]) -> State:
        cur_piece = self.configuration[move[0][0]][move[0][1]]
        configuration: List[List[Piece]] = copy.deepcopy(self.configuration)
        if not evaluate_move(move, cur_piece):
            return False
        configuration[move[1][0]][move[1][1]] = cur_piece
        return State()

        
