from flask import Flask, request, jsonify, render_template
import chess
from maia2 import model, inference
import random
import numpy as np

app = Flask(__name__)

# Global game state
board = chess.Board()
bot_elo = 900
player_elo_estimate = 1050
initial_player_elo = 1050  # Rating selected at game start
move_history = []  # Store player's move evaluations
test_ratings = [1000, 1200, 1400, 1600, 1800, 2000]

# Initialize Maia 2 model
print("Loading Maia 2 model...")
m = model.from_pretrained(type="rapid", device="cpu")  # Use "gpu" if available
prepared = inference.prepare()
print("Model loaded successfully!")

"""
Analyze a player's move by checking how likely it is at different rating levels.
Returns estimated rating for this single move.
"""
def estimate_player_rating_from_move(fen_before_move, player_move_uci):

    move_scores = {}

    for test_elo in test_ratings:
        # Get move probabilities as if player were at this rating
        probs, _ = inference.inference_each(
            m, prepared, fen_before_move,
            elo_self=test_elo,
            elo_oppo=bot_elo
        )

        # Get probability of the actual move the player made
        move_prob = probs.get(player_move_uci, 0.0)
        move_scores[test_elo] = move_prob

    print(f"Move {player_move_uci} probabilities at different ratings: {move_scores}")

    # Find the rating where this move is most likely
    if move_scores is None:
        print(f'Error. No data found. Returning 1100')
        return 1100, move_scores  # Default if no data


    min_prob = min(move_scores.values())
    max_prob = max(move_scores.values())
    discriminative_power = max_prob - min_prob

    if all(p == 0 for p in move_scores.values()) or discriminative_power < 0.2:
        print(f'Unlikely or likely move for each score. Making weighted_rating closer to bot')
        weighted_rating = int (bot_elo + player_elo_estimate) / 2
        return weighted_rating, move_scores


    top2 = sorted(
        move_scores.items(),
        key=lambda x: (-x[1], x[0])
    )[:2]

    weighted_rating = (top2[0][0] + top2[1][0]) // 2
    print(f"Weighted rating {weighted_rating}")
    return weighted_rating, move_scores




"""
Calculate player's estimated rating based on all moves played so far.
Recent moves are weighted more heavily.
"""
def get_cumulative_player_rating():
    if not move_history:
        return 1100

    # Weight recent moves more heavily (exponential decay)
    total_weight = 0
    weighted_sum = 0

    for i, (rating, _) in enumerate(move_history):
        # More recent moves get higher weight
        weight = np.exp(i / len(move_history))
        weighted_sum += rating * weight
        total_weight += weight

    cumulative_rating = int(weighted_sum / total_weight)

    print(f"Cumulative rating from {len(move_history)} moves: {cumulative_rating}")
    return cumulative_rating


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/state")
def state():
    """Get current game state"""
    return jsonify({
        "fen": board.fen(),
        "bot_elo": bot_elo,
        "player_elo": player_elo_estimate,
        "legal_moves": [move.uci() for move in board.legal_moves],
        "moves_analyzed": len(move_history)
    })


@app.route("/move", methods=["POST"])
def move():
    global bot_elo, player_elo_estimate, move_history

    try:
        player_move = request.json["move"]

        # Store FEN before player's move for analysis
        fen_before = board.fen()

        # Validate and make player's move
        try:
            move_obj = chess.Move.from_uci(player_move)
            if move_obj not in board.legal_moves:
                return jsonify({"error": "Illegal move"}), 400
            board.push(move_obj)
        except Exception as e:
            return jsonify({"error": f"Invalid move format: {str(e)}"}), 400

        # Analyze player's move at different rating levels
        print(f"\n--- Analyzing player move: {player_move} ---")

        # For the first move, use the initial rating selected at game start
        if len(move_history) == 0:
            print(f"First move - using initial rating: {initial_player_elo}")
            move_rating = initial_player_elo
            move_scores = {elo: 0.0 for elo in test_ratings}  # Placeholder
        else:
            move_rating, move_scores = estimate_player_rating_from_move(fen_before, player_move)

        move_history.append((move_rating, move_scores))

        # Get cumulative rating estimate
        player_elo_estimate = get_cumulative_player_rating()

        print(f"This move suggests rating: {move_rating}")
        print(f"Cumulative estimated rating: {player_elo_estimate}")

        # Get win probability after player's move (from player's perspective)

        medium_rating = (player_elo_estimate + bot_elo) / 2
        _, player_win_prob = inference.inference_each(
            m, prepared, board.fen(),
            elo_self=medium_rating,
            elo_oppo=medium_rating
        )
        print(f"Player win probability after their move: {player_win_prob}")

        # Check if game is over after player's move
        if board.is_game_over():
            return jsonify({
                "bot_move": None,
                "fen": board.fen(),
                "bot_elo": bot_elo,
                "player_elo": player_elo_estimate,
                "move_rating": move_rating,
                "move_scores": move_scores,
                "player_win_prob": player_win_prob,
                "game_over": True,
                "result": board.result()
            })

        # Adjust bot rating slowly toward player rating (but stay weaker)
        win_delta = -200
        match True:
            case _ if player_win_prob > 0.85:
                win_delta = 100
            case _ if player_win_prob > 0.75:
                win_delta = 50
            case _ if player_win_prob > 0.65:
                win_delta = 0
            case _ if player_win_prob > 0.5:
                win_delta = -100


        target_bot_elo = player_elo_estimate - 150 + win_delta
        bot_elo = (target_bot_elo + bot_elo) / 2

        print(f"Bot rating adjusted: {bot_elo} (target: {target_bot_elo})")

        # Get bot's move using Maia 2 at bot's current rating
        try:
            print(f"\n--- Bot's turn (Rating: {bot_elo}) ---")
            print(f"Current FEN being sent to Maia: {board.fen()}")
            print(f"Turn to move: {'White' if board.turn else 'Black'}")
            print(f"Move number: {board.fullmove_number}")

            # Print last few moves for context
            if len(board.move_stack) > 0:
                print(f"Last moves:")
                for i, move in enumerate(board.move_stack[-4:]):
                    print(f"  {i + 1}. {move.uci()}")

            probs, maia_info = inference.inference_each(
                m, prepared, board.fen(),
                elo_self=bot_elo,
                elo_oppo=player_elo_estimate
            )

            print(f"Maia returned info: {maia_info}")
            print(f"Maia returned {len(probs)} possible moves")
            print(f"Top 5 moves with probabilities:")
            sorted_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for move, prob in sorted_moves:
                # Parse move to understand what piece is moving where
                from_sq = move[:2]
                to_sq = move[2:4]
                piece_on_from = board.piece_at(chess.parse_square(from_sq))
                print(f"  {move}: {prob:.4f} (moving {piece_on_from} from {from_sq} to {to_sq})")

            # Select the most probable move

            moves = [mv for mv, p in sorted_moves]
            weights = [p for mv, p in sorted_moves]

            # Выбор хода с учётом вероятностей
            bot_move_uci = random.choices(moves, weights=weights, k=1)[0]
            selected_prob = dict(sorted_moves)[bot_move_uci]

            print(f"\nBot selected: {bot_move_uci} (probability: {selected_prob:.4f})")

            # Make bot's move
            try:
                board.push_uci(bot_move_uci)
            except Exception as e:
                print(f"Error pushing move {bot_move_uci}: {e}")
                # Fallback: pick first legal move
                bot_move_uci = list(board.legal_moves)[0].uci()
                board.push_uci(bot_move_uci)
                print(f"Fallback to: {bot_move_uci}")

            medium_rating = (player_elo_estimate+ bot_elo) / 2
            # Get win probability after bot's move (from player's perspective)
            if not board.is_game_over():
                _, player_win_prob_after_bot = inference.inference_each(
                    m, prepared, board.fen(),
                    elo_self=medium_rating,
                    elo_oppo=medium_rating
                )
            else:
                player_win_prob_after_bot = player_win_prob
            print(f"Player win probability after bot's move: {player_win_prob_after_bot}")

            return jsonify({
                "bot_move": bot_move_uci,
                "fen": board.fen(),
                "bot_elo": bot_elo,
                "player_elo": player_elo_estimate,
                "move_rating": move_rating,
                "move_scores": move_scores,
                "player_win_prob": player_win_prob_after_bot,
                "game_over": board.is_game_over(),
                "result": board.result() if board.is_game_over() else None
            })

        except Exception as e:
            return jsonify({"error": f"Bot error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/set_initial_rating", methods=["POST"])
def set_initial_rating():
    global bot_elo, player_elo_estimate, initial_player_elo
    player_elo_estimate = request.json.get("player_elo", 1100)
    initial_player_elo = player_elo_estimate  # Save for first move
    bot_elo = player_elo_estimate - 150
    print(f"Initial bot rating set to: {bot_elo}")
    print(f"Initial player rating estimate set to: {player_elo_estimate}")
    return jsonify({
        "bot_elo": bot_elo,
        "player_elo": player_elo_estimate
    })


@app.route("/reset", methods=["POST"])
def reset():
    global board, bot_elo, player_elo_estimate, initial_player_elo, move_history
    board = chess.Board()
    player_elo_estimate = 1150  # Reset player estimate
    initial_player_elo = 1150  # Reset initial rating
    bot_elo = 1000  # Reset to starting difficulty
    move_history = []  # Clear move history
    return jsonify({
        "fen": board.fen(),
        "bot_elo": bot_elo,
        "player_elo": player_elo_estimate
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)



