import chess
import chess.engine
import chess.pgn # For PGN export
import torch
import os
import time
import random
import math

# --- Configuration Imports (from your project's config.py) ---
# Assuming DEVICE and NUM_SIMULATIONS (for your engine) might be used or referenced.
from config import DEVICE, NUM_SIMULATIONS as YOUR_ENGINE_DEFAULT_SIMULATIONS

# --- Project Module Imports ---
from neural_net import ChessNN # Your Neural Network class
from mcts import MCTS         # Your MCTS class
# No direct utils imports needed here usually

# --- Your Engine Wrapper ---
class AlphaChessEngine:
    def __init__(self, model_path, num_simulations):
        self.model = self._load_nn_model(model_path, ChessNN)
        self.num_simulations = num_simulations
        self.mcts_instance = None

    def _load_nn_model(self, model_path, model_class):
        model = model_class().to(DEVICE)
        if os.path.exists(model_path):
            print(f"Loading AlphaChess model from {model_path}")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else:
            raise FileNotFoundError(f"AlphaChess model not found at {model_path}")
        return model

    def play(self, board: chess.Board, time_limit_ms=None):
        if self.mcts_instance is None or self.mcts_instance.root_board_fen != board.fen():
            self.mcts_instance = MCTS(self.model, board.copy())

        _policy_dist, best_move = self.mcts_instance.search(board.copy(), self.num_simulations)
        return best_move

# --- Match Orchestrator ---
def play_match(engine1_play_func, engine1_name,
               engine2_uci_engine_obj, engine2_name,
               engine2_uci_config_options, # Dictionary for UCI limits
               num_games_to_play,
               time_limit_per_move_ms_for_engine1=1000,
               pgn_save_directory="elo_match_pgns"): # Default directory name

    scores = {engine1_name: 0, engine2_name: 0, "draws": 0}
    # Ensure the directory exists, create if not
    os.makedirs(pgn_save_directory, exist_ok=True)
    print(f"Saving PGNs to directory: {os.path.abspath(pgn_save_directory)}") # Show full path

    for i in range(num_games_to_play):
        current_board = chess.Board()
        pgn_game_obj = chess.pgn.Game()
        pgn_game_obj.headers["Event"] = "AI Evaluation Match"
        pgn_game_obj.headers["Site"] = "Local Execution"
        pgn_game_obj.headers["Date"] = time.strftime("%Y.%m.%d")
        pgn_game_obj.headers["Round"] = str(i + 1)

        current_pgn_node = pgn_game_obj

        # Assign players for this game
        if i % 2 == 0: # Engine1 (Your AI) plays White
            white_engine_handler, white_player_name = engine1_play_func, engine1_name
            black_engine_handler, black_player_name = engine2_uci_engine_obj, engine2_name
        else: # Engine1 (Your AI) plays Black
            white_engine_handler, white_player_name = engine2_uci_engine_obj, engine2_name
            black_engine_handler, black_player_name = engine1_play_func, engine1_name

        pgn_game_obj.headers["White"] = white_player_name
        pgn_game_obj.headers["Black"] = black_player_name

        print(f"\nGame {i+1}/{num_games_to_play}: {white_player_name} (White) vs {black_player_name} (Black)")

        move_counter = 0
        game_error = False # Flag to track if game ended abnormally
        while not current_board.is_game_over(claim_draw=True):
            move_counter += 1
            active_engine_handler = white_engine_handler if current_board.turn == chess.WHITE else black_engine_handler
            active_player_name = white_player_name if current_board.turn == chess.WHITE else black_player_name

            generated_move = None
            try:
                if active_engine_handler == engine2_uci_engine_obj: # External UCI Engine (Stockfish)
                    uci_limit = chess.engine.Limit() # Default empty limit
                    if "depth" in engine2_uci_config_options:
                        uci_limit = chess.engine.Limit(depth=engine2_uci_config_options["depth"])
                    elif "nodes" in engine2_uci_config_options:
                        uci_limit = chess.engine.Limit(nodes=engine2_uci_config_options["nodes"])
                    elif "movetime" in engine2_uci_config_options: # movetime in ms
                         uci_limit = chess.engine.Limit(time=engine2_uci_config_options["movetime"] / 1000.0)

                    engine_play_result = active_engine_handler.play(current_board, uci_limit)
                    generated_move = engine_play_result.move
                else: # Your AI's turn (engine1_play_func)
                    generated_move = active_engine_handler(current_board.copy(), time_limit_per_move_ms_for_engine1)

            except chess.engine.EngineTerminatedError:
                print(f"ERROR: UCI Engine {active_player_name} terminated unexpectedly.")
                pgn_game_obj.headers["Termination"] = "Engine Crash"
                game_error = True
                break
            except Exception as e:
                print(f"ERROR during move generation for {active_player_name}: {e}")
                pgn_game_obj.headers["Termination"] = "Move Generation Error"
                game_error = True
                break

            if generated_move is None or not current_board.is_legal(generated_move):
                error_msg = "no move" if generated_move is None else f"illegal move ({generated_move.uci()})"
                print(f"CRITICAL ERROR: {active_player_name} returned {error_msg}. Board:\n{current_board}")
                pgn_game_obj.headers["Termination"] = f"{active_player_name} Forfeits ({error_msg})"
                game_error = True
                # Assign loss to the player who failed to move
                if active_player_name == white_player_name: # White failed
                    pgn_game_obj.headers["Result"] = "0-1"
                    scores[black_player_name] +=1
                else: # Black failed
                    pgn_game_obj.headers["Result"] = "1-0"
                    scores[white_player_name] +=1
                break # End this game

            # Add move to PGN and board
            current_pgn_node = current_pgn_node.add_variation(generated_move)
            current_board.push(generated_move)

        # After game loop finishes (game over or error)
        if not game_error: # Only process normal outcome if no error occurred
            game_outcome_obj = current_board.outcome(claim_draw=True)
            if game_outcome_obj:
                pgn_game_obj.headers["Result"] = game_outcome_obj.result()
                pgn_game_obj.headers["Termination"] = game_outcome_obj.termination.name.title()
                print(f"Game {i+1} official result: {game_outcome_obj.result()} ({game_outcome_obj.termination.name})")
                if game_outcome_obj.winner == chess.WHITE:
                    scores[white_player_name] += 1
                elif game_outcome_obj.winner == chess.BLACK:
                    scores[black_player_name] += 1
                else: # Draw
                    scores["draws"] += 1
            else:
                pgn_game_obj.headers["Result"] = "*"
                pgn_game_obj.headers["Termination"] = "Incomplete"
                print(f"Game {i+1} ended inconclusively (no outcome object).")

        # Save the PGN file regardless of how the game ended
        pgn_file_name = os.path.join(pgn_save_directory, f"game_{i+1}_{white_player_name}_vs_{black_player_name}.pgn")
        try:
            with open(pgn_file_name, "w", encoding="utf-8") as pgn_file:
                exporter = chess.pgn.FileExporter(pgn_file)
                pgn_game_obj.accept(exporter)
            print(f"PGN saved to {pgn_file_name}")
        except Exception as e:
            print(f"Error saving PGN {pgn_file_name}: {e}")

        print(f"Current scores: {engine1_name}: {scores.get(engine1_name, 0)}, {engine2_name}: {scores.get(engine2_name, 0)}, Draws: {scores.get('draws', 0)}")

    return scores


# --- ELO Difference Calculation ---
def elo_diff(score_points, total_games_in_match):
    """ Calculates Elo difference based on score = wins + 0.5 * draws """
    if total_games_in_match == 0: return 0.0
    # Avoid extreme ratios which cause issues with log
    if score_points <= 0.001 * total_games_in_match: return -800.0
    if score_points >= 0.999 * total_games_in_match: return 800.0

    expected_score_ratio = score_points / total_games_in_match
    # Standard ELO difference formula: D = -400 * log10(1/P - 1)
    return -400.0 * math.log10(1.0 / expected_score_ratio - 1.0)


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- User Configurations ---
    YOUR_MODEL_PATH = "./checkpoints_chess/latest_model.pth"
    # YOUR_MODEL_PATH = "./checkpoints_chess/model_iter_1.pth"
    # !!! IMPORTANT: SET THIS TO YOUR ACTUAL STOCKFISH PATH !!!
    STOCKFISH_ENGINE_PATH = r"./engines/stockfish/stockfish-windows-x86-64-avx2.exe" # Example for Windows, relative path
    # STOCKFISH_ENGINE_PATH = r"C:\Path\To\Your\stockfish.exe" # Example absolute path
    # STOCKFISH_ENGINE_PATH = "/path/to/your/stockfish" # Example for Linux

    NUM_GAMES_VS_STOCKFISH = 4       # Number of games
    YOUR_ENGINE_SIMULATIONS = 100     # MCTS sims for your AI
    YOUR_ENGINE_TIME_LIMIT_MS = 500  # Time per move for your AI (ms)

    # Configuration for the Stockfish opponent
    stockfish_opponent_config = {
        "Skill Level": 0,
        "depth": 1,
        "movetime": 1 # Very weak Stockfish (1ms per move)
    }
    # Example alternative: {"Skill Level": 1, "movetime": 250}
    # Example alternative: {"Skill Level": 0, "depth": 2}

    PGN_SAVE_DIR = "./elo_match_pgns" # Directory where PGNs will be saved (relative to script location)

    # --- Estimated ELOs for Stockfish Configs (Very Rough Guide) ---
    stockfish_base_elos_by_skill = {
        0: 900, 1: 1050, 2: 1200, 3: 1350, 4: 1500, 5: 1650,
        10: 2000, 15: 2500, 20: 3000
    }

    # --- Script Logic ---
    if not os.path.exists(YOUR_MODEL_PATH):
        print(f"ERROR: Your AI model not found at '{YOUR_MODEL_PATH}'. Please check the path.")
        exit()

    if not os.path.exists(STOCKFISH_ENGINE_PATH):
        print(f"ERROR: Stockfish engine not found at '{STOCKFISH_ENGINE_PATH}'.")
        print("Please download Stockfish, place it correctly, and update STOCKFISH_ENGINE_PATH in this script.")
        exit()

    print("Initializing your AlphaChess Engine...")
    alpha_chess_ai_player = AlphaChessEngine(YOUR_MODEL_PATH, YOUR_ENGINE_SIMULATIONS)

    print(f"Initializing Stockfish (Config: {stockfish_opponent_config})...")
    stockfish_process = None
    try:
        stockfish_process = chess.engine.SimpleEngine.popen_uci(STOCKFISH_ENGINE_PATH)
        if "Skill Level" in stockfish_opponent_config:
             stockfish_process.configure({"Skill Level": stockfish_opponent_config["Skill Level"]})
    except Exception as e:
        print(f"ERROR: Failed to start or configure Stockfish engine: {e}")
        if stockfish_process: stockfish_process.quit()
        exit()

    print(f"\nStarting ELO estimation match: AlphaChess vs Stockfish")
    print(f"Number of games: {NUM_GAMES_VS_STOCKFISH}")
    print(f"Your AI (AlphaChess): {YOUR_ENGINE_SIMULATIONS} simulations, {YOUR_ENGINE_TIME_LIMIT_MS}ms/move")
    print(f"Opponent (Stockfish): Config = {stockfish_opponent_config}")
    print(f"PGNs will be saved to: {PGN_SAVE_DIR}")

    match_results = play_match(
        engine1_play_func=alpha_chess_ai_player.play,
        engine1_name="AlphaChess",
        engine2_uci_engine_obj=stockfish_process,
        engine2_name="Stockfish",
        engine2_uci_config_options=stockfish_opponent_config,
        num_games_to_play=NUM_GAMES_VS_STOCKFISH,
        time_limit_per_move_ms_for_engine1=YOUR_ENGINE_TIME_LIMIT_MS,
        pgn_save_directory=PGN_SAVE_DIR
    )

    # --- ELO Calculation & Display ---
    print("\n--- Overall Match Results ---")
    your_wins = match_results.get('AlphaChess', 0)
    opponent_wins = match_results.get('Stockfish', 0)
    draws = match_results.get('draws', 0)
    print(f"AlphaChess Wins: {your_wins}")
    print(f"Stockfish Wins: {opponent_wins}")
    print(f"Draws: {draws}")

    # Estimate opponent ELO based on config (rough guide)
    sf_skill = stockfish_opponent_config.get("Skill Level", 0)
    estimated_opponent_elo = stockfish_base_elos_by_skill.get(sf_skill, 800) # Default ELO if skill unknown

    # Adjust based on time/depth limits
    if "movetime" in stockfish_opponent_config:
        if stockfish_opponent_config["movetime"] <= 1: estimated_opponent_elo = min(estimated_opponent_elo, 400) # 1ms is very weak
        elif stockfish_opponent_config["movetime"] <= 10: estimated_opponent_elo = min(estimated_opponent_elo, 600)
        elif stockfish_opponent_config["movetime"] <= 50: estimated_opponent_elo = min(estimated_opponent_elo, 750)
        elif stockfish_opponent_config["movetime"] <= 100: estimated_opponent_elo = min(estimated_opponent_elo, 850)
    elif "depth" in stockfish_opponent_config:
        if stockfish_opponent_config["depth"] == 1: estimated_opponent_elo = min(estimated_opponent_elo, 600)
        elif stockfish_opponent_config["depth"] <= 3: estimated_opponent_elo = min(estimated_opponent_elo, 800)

    print(f"Using estimated ELO for Stockfish (Config: {stockfish_opponent_config}): ~{estimated_opponent_elo}")

    # Calculate ELO for your AI
    your_ai_score_points = your_wins + 0.5 * draws
    total_games_in_match = your_wins + opponent_wins + draws

    if total_games_in_match > 0:
        elo_difference_from_opponent = elo_diff(your_ai_score_points, total_games_in_match)
        your_ai_estimated_elo = estimated_opponent_elo + elo_difference_from_opponent
        print(f"Your AI's score: {your_ai_score_points} / {total_games_in_match}")
        print(f"ELO difference from opponent: {elo_difference_from_opponent:.0f}")
        print(f"Your AlphaChess AI Estimated ELO: ~{your_ai_estimated_elo:.0f}")
    else:
        print("No games were completed, cannot estimate ELO.")

    # --- Cleanup: Close Stockfish Process ---
    if stockfish_process:
        stockfish_process.quit()
    print("\nELO estimation match finished.")