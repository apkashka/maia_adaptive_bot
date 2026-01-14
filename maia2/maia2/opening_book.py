"""
Opening book module to handle chess openings using local Polyglot books.
This prevents the repetition issue in early game positions (first 10 halfmoves).
Uses different books based on player skill level (ELO).
"""

import os
import chess
import chess.polyglot
from typing import Dict, Optional


# Configuration: Opening books for different skill levels
OPENING_BOOKS_CONFIG = {
    'weak_medium': 'maia2/maia2/opening_books/Human.bin',        # ELO 800-1600
    'medium': 'maia2/maia2/opening_books/rodent.bin',       # ELO 1600-1900
    'strong': 'maia2/maia2/opening_books/Titans.bin',              # ELO 1900-2400+
}

def get_opening_book_path(elo: int) -> str:
    """
    Select appropriate opening book based on player ELO.

    Args:
        elo: Player's ELO rating

    Returns:
        Path to the appropriate opening book file
    """
    if elo < 1500:
        print('human opening selected')
        return OPENING_BOOKS_CONFIG['weak_medium']
    elif elo < 1900:
        print('medium opening selected')
        return OPENING_BOOKS_CONFIG['medium']
    else:
        print('hardcode opening selected')
        return OPENING_BOOKS_CONFIG['strong']

#todo just count ourselves
def get_halfmove_count(fen: str) -> int:
    """
    Extract the halfmove count from a FEN string.

    Args:
        fen: FEN string representation of the position

    Returns:
        Number of half-moves (plies) played from the starting position
    """
    try:
        # FEN format: position active_color castling en_passant halfmove_clock fullmove_number
        parts = fen.split(' ')
        if len(parts) >= 6:
            fullmove_number = int(parts[5])
            active_color = parts[1]
            # Calculate halfmoves: (fullmove - 1) * 2 + (1 if black to move else 0)
            halfmoves = (fullmove_number - 1) * 2
            if active_color == 'b':
                halfmoves += 1
            return halfmoves
        return 0
    except (ValueError, IndexError):
        return 0


def fetch_opening_book_moves(fen: str, elo: int = 1500) -> Dict[str, float]:
    """
    Fetch opening book moves from local Polyglot book based on player ELO.

    Args:
        fen: FEN string of the current position
        elo: Player's ELO rating (default: 1500)

    Returns:
        Dictionary mapping UCI moves to their probabilities (normalized)
        Returns empty dict if book not found or no moves available
    """
    try:
        # Select book based on ELO
        book_path = get_opening_book_path(elo)
        # Check if book file exists
        if not os.path.exists(book_path):
            print(f"Warning: Opening book not found at {book_path}")
            return {}

        # Parse FEN to board
        board = chess.Board(fen)

        # Read moves from Polyglot book
        move_weights = {}

        with chess.polyglot.open_reader(book_path) as reader:
            for entry in reader.find_all(board):
                move_uci = entry.move.uci()
                weight = entry.weight
                move_weights[move_uci] = weight

        if not move_weights:
            return {}

        # Normalize weights to probabilities
        total_weight = sum(move_weights.values())
        if total_weight == 0:
            return {}

        move_probs = {}
        for move_uci, weight in move_weights.items():
            probability = weight / total_weight
            move_probs[move_uci] = round(probability, 4)

        # Sort by probability (descending)
        move_probs = dict(sorted(move_probs.items(), key=lambda item: item[1], reverse=True))

        return move_probs

    except Exception as e:
        # If anything goes wrong, return empty dict to fall back to neural network
        print(f"Warning: Opening book fetch failed: {e}")
        return {}


def should_use_opening_book(fen: str, max_halfmoves: int = 10) -> bool:
    """
    Determine if opening book should be used for this position.

    Args:
        fen: FEN string of the current position
        max_halfmoves: Maximum number of half-moves to use opening book for

    Returns:
        True if opening book should be used, False otherwise
    """
    halfmoves = get_halfmove_count(fen)
    return halfmoves < max_halfmoves


def get_opening_book_or_fallback(fen: str, model_probs: Dict[str, float],
                                 elo_self: int = 1500, max_halfmoves: int = 10) -> tuple[Dict[str, float], bool]:
    """
    Get opening book moves if available, otherwise return model predictions.

    Args:
        fen: FEN string of the current position
        model_probs: Model's predicted move probabilities (fallback)
        elo_self: Player's ELO rating (default: 1500)
        max_halfmoves: Maximum number of half-moves to use opening book for

    Returns:
        Tuple of (move_probabilities_dict, used_opening_book_flag)
    """
    if should_use_opening_book(fen, max_halfmoves):
        opening_moves = fetch_opening_book_moves(fen, elo=elo_self)
        if opening_moves:
            return opening_moves, True

    return model_probs, False