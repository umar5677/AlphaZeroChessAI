import pygame
import chess
import sys
import os
import torch

from config import DEVICE, NUM_SIMULATIONS
from neural_net import ChessNN
from mcts import MCTS
from utils import board_to_tensor

# --- Init model and MCTS ---
model_path = "./checkpoints_chess/latest_model.pth"
model = ChessNN().to(DEVICE)
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
else:
    print("Warning: model not found. Using untrained model.")
    model.eval()

mcts = MCTS(model, chess.Board())  # dummy init board
board = chess.Board()

# --- Pygame setup ---
WIDTH, HEIGHT = 640, 640
SQ_SIZE = WIDTH // 8

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess GUI")

# Load pieces
pieces = {}
for piece in ['wP','wR','wN','wB','wQ','wK','bP','bR','bN','bB','bQ','bK']:
    pieces[piece] = pygame.transform.scale(pygame.image.load(f'assets/{piece}.png'), (SQ_SIZE, SQ_SIZE))

# --- Draw Board ---
def draw_board():
    colors = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            key = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
            screen.blit(pieces[key], pygame.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Convert mouse to square
def get_square_under_mouse(pos):
    col = pos[0] // SQ_SIZE
    row = pos[1] // SQ_SIZE
    return chess.square(col, 7 - row)

# --- Main loop ---
def main():
    selected_square = None
    running = True
    player_color = chess.WHITE  # or allow switch

    while running:
        draw_board()
        draw_pieces()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if board.turn == player_color and event.type == pygame.MOUSEBUTTONDOWN:
                square = get_square_under_mouse(pygame.mouse.get_pos())

                if selected_square is None:
                    # First click
                    if board.piece_at(square) and board.piece_at(square).color == player_color:
                        selected_square = square
                else:
                    # Second click
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                        print(f"You played: {move.uci()}")

                        # --- AI MOVE ---
                        if not board.is_game_over():
                            print("AI thinking...")
                            _, ai_move = mcts.search(board.copy(), NUM_SIMULATIONS)
                            if ai_move and ai_move in board.legal_moves:
                                board.push(ai_move)
                                print(f"AI plays: {ai_move.uci()}")
                    else:
                        print("Illegal move.")
                        selected_square = None

        if board.is_game_over():
            print("Game Over:", board.result())
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
