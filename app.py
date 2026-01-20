from flask import Flask, request, jsonify, render_template
import chess
from maia2 import model, inference
import random
import numpy as np
import uuid
import time

app = Flask(__name__)

# Game sessions storage
games = {}  # {game_id: {board, bot_elo, player_elo, ...}}
SESSION_TIMEOUT = 1800  # 30 minutes

test_ratings = [1000, 1200, 1400, 1600, 1800, 2000]

# Initialize Maia 2 model (shared across all games)
print("Loading Maia 2 model...")
m = model.from_pretrained(type="rapid", device="cpu")
prepared = inference.prepare()
print("Model loaded successfully!")


def create_game(initial_elo=1100):
    """Create a new game session"""
    game_id = str(uuid.uuid4())[:8]
    games[game_id] = {
        'board': chess.Board(),
        'bot_elo': initial_elo - 150,
        'player_elo': initial_elo,
        'initial_player_elo': initial_elo,
        'move_history': [],
        'last_activity': time.time()
    }
    print(f"Created new game: {game_id}")
    return game_id


def get_game(game_id):
    """Get game by ID, return None if not found or expired"""
    if game_id not in games:
        return None

    game = games[game_id]
    game['last_activity'] = time.time()
    return game


def cleanup_old_games():
    """Remove games inactive for more than SESSION_TIMEOUT"""
    now = time.time()
    expired = [gid for gid, g in games.items()
               if now - g['last_activity'] > SESSION_TIMEOUT]
    for gid in expired:
        del games[gid]
        print(f"Cleaned up expired game: {gid}")

    if expired:
        print(f"Active games: {len(games)}")


def estimate_player_rating_from_move(game, fen_before_move, player_move_uci):
    """
    Analyze a player's move by checking how likely it is at different rating levels.
    Returns estimated rating for this single move.
    """
    move_scores = {}

    for test_elo in test_ratings:
        probs, _ = inference.inference_each(
            m, prepared, fen_before_move,
            elo_self=test_elo,
            elo_oppo=game['bot_elo'],
            use_opening_book=False  # Use raw Maia model for rating estimation
        )
        move_prob = probs.get(player_move_uci, 0.0)
        move_scores[test_elo] = move_prob

    print(f"Move {player_move_uci} probabilities at different ratings: {move_scores}")

    if move_scores is None:
        print(f'Error. No data found. Returning 1100')
        return 1100, move_scores

    min_prob = min(move_scores.values())
    max_prob = max(move_scores.values())
    discriminative_power = max_prob - min_prob

    if all(p == 0 for p in move_scores.values()) or discriminative_power < 0.2:
        print(f'Unlikely or likely move for each score. Making weighted_rating closer to bot')
        weighted_rating = int((game['bot_elo'] + game['player_elo']) / 2)
        return weighted_rating, move_scores

    top2 = sorted(
        move_scores.items(),
        key=lambda x: (-x[1], x[0])
    )[:2]

    weighted_rating = (top2[0][0] + top2[1][0]) // 2
    print(f"Weighted rating {weighted_rating}")
    return weighted_rating, move_scores


def get_cumulative_player_rating(game):
    """
    Calculate player's estimated rating based on all moves played so far.
    Recent moves are weighted more heavily.
    """
    move_history = game['move_history']

    if not move_history:
        return 1100

    total_weight = 0
    weighted_sum = 0

    for i, (rating, _) in enumerate(move_history):
        weight = np.exp(i / len(move_history))
        weighted_sum += rating * weight
        total_weight += weight

    cumulative_rating = int(weighted_sum / total_weight)
    print(f"Cumulative rating from {len(move_history)} moves: {cumulative_rating}")
    return cumulative_rating


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/new_game", methods=["POST"])
def new_game():
    """Create a new game session"""
    cleanup_old_games()

    initial_elo = request.json.get("player_elo", 1100)
    game_id = create_game(initial_elo)
    game = games[game_id]

    return jsonify({
        "game_id": game_id,
        "fen": game['board'].fen(),
        "bot_elo": game['bot_elo'],
        "player_elo": game['player_elo']
    })


@app.route("/state", methods=["POST"])
def state():
    """Get current game state"""
    game_id = request.json.get("game_id")
    game = get_game(game_id)

    if not game:
        return jsonify({"error": "Game not found"}), 404

    board = game['board']
    return jsonify({
        "fen": board.fen(),
        "bot_elo": game['bot_elo'],
        "player_elo": game['player_elo'],
        "legal_moves": [move.uci() for move in board.legal_moves],
        "moves_analyzed": len(game['move_history'])
    })


@app.route("/move", methods=["POST"])
def move():
    """Process player move and get bot response"""
    game_id = request.json.get("game_id")
    game = get_game(game_id)

    if not game:
        return jsonify({"error": "Game not found"}), 404

    try:
        player_move = request.json["move"]
        board = game['board']

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
        print(f"\n--- Game {game_id}: Analyzing player move: {player_move} ---")

        # For the first move, use the initial rating selected at game start
        if len(game['move_history']) == 0:
            print(f"First move - using initial rating: {game['initial_player_elo']}")
            move_rating = game['initial_player_elo']
            move_scores = {elo: 0.0 for elo in test_ratings}
        else:
            move_rating, move_scores = estimate_player_rating_from_move(game, fen_before, player_move)

        game['move_history'].append((move_rating, move_scores))

        # Get cumulative rating estimate
        game['player_elo'] = get_cumulative_player_rating(game)

        print(f"This move suggests rating: {move_rating}")
        print(f"Cumulative estimated rating: {game['player_elo']}")

        # Get win probability after player's move
        medium_rating = (game['player_elo'] + game['bot_elo']) / 2
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
                "bot_elo": game['bot_elo'],
                "player_elo": game['player_elo'],
                "move_rating": move_rating,
                "move_scores": move_scores,
                "player_win_prob": player_win_prob,
                "game_over": True,
                "result": board.result()
            })

        # Adjust bot rating based on win probability
        win_delta = -200
        match True:
            case _ if player_win_prob > 0.8:
                win_delta = 100
            case _ if player_win_prob > 0.7:
                win_delta = 50
            case _ if player_win_prob > 0.6:
                win_delta = -100
            case _ if player_win_prob > 0.5:
                win_delta = -150

        target_bot_elo = game['player_elo'] + win_delta
        game['bot_elo'] = (target_bot_elo + game['bot_elo']) / 2

        print(f"Bot rating adjusted: {game['bot_elo']} (target: {target_bot_elo})")

        # Get bot's move
        try:
            print(f"\n--- Bot's turn (Rating: {game['bot_elo']}) ---")

            probs, maia_info = inference.inference_each(
                m, prepared, board.fen(),
                elo_self=game['bot_elo'],
                elo_oppo=game['player_elo']
            )

            sorted_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

            print(f"Top 3 moves:")
            for mv, prob in sorted_moves:
                print(f"  {mv}: {prob:.4f}")

            moves = [mv for mv, p in sorted_moves]
            weights = [p for mv, p in sorted_moves]

            bot_move_uci = random.choices(moves, weights=weights, k=1)[0]
            selected_prob = dict(sorted_moves)[bot_move_uci]

            print(f"Bot selected: {bot_move_uci} (probability: {selected_prob:.4f})")

            # Make bot's move
            try:
                board.push_uci(bot_move_uci)
            except Exception as e:
                print(f"Error pushing move {bot_move_uci}: {e}")
                bot_move_uci = list(board.legal_moves)[0].uci()
                board.push_uci(bot_move_uci)

            # Get win probability after bot's move
            if not board.is_game_over():
                medium_rating = (game['player_elo'] + game['bot_elo']) / 2
                _, player_win_prob_after_bot = inference.inference_each(
                    m, prepared, board.fen(),
                    elo_self=medium_rating,
                    elo_oppo=medium_rating
                )
            else:
                player_win_prob_after_bot = player_win_prob

            return jsonify({
                "bot_move": bot_move_uci,
                "fen": board.fen(),
                "bot_elo": game['bot_elo'],
                "player_elo": game['player_elo'],
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


@app.route("/reset", methods=["POST"])
def reset():
    """Reset current game"""
    game_id = request.json.get("game_id")
    game = get_game(game_id)

    if not game:
        return jsonify({"error": "Game not found"}), 404

    initial_elo = game['initial_player_elo']
    game['board'] = chess.Board()
    game['player_elo'] = initial_elo
    game['bot_elo'] = initial_elo - 150
    game['move_history'] = []

    return jsonify({
        "fen": game['board'].fen(),
        "bot_elo": game['bot_elo'],
        "player_elo": game['player_elo']
    })


@app.route("/stats")
def stats():
    """Get server stats"""
    cleanup_old_games()
    return jsonify({
        "active_games": len(games),
        "game_ids": list(games.keys())
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
