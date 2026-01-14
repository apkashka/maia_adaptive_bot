"""
Test script to verify the opening book integration works correctly.
"""
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from maia2.opening_book import (
    get_halfmove_count,
    fetch_opening_book_moves,
    should_use_opening_book,
    get_opening_book_or_fallback
)


def test_halfmove_count():
    """Test halfmove counting from FEN strings"""
    print("Testing halfmove count...")

    # Starting position
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert get_halfmove_count(fen1) == 0, "Starting position should be 0 halfmoves"
    print(f"  ✓ Starting position: {get_halfmove_count(fen1)} halfmoves")

    # After 1. e4
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert get_halfmove_count(fen2) == 1, "After 1. e4 should be 1 halfmove"
    print(f"  ✓ After 1. e4: {get_halfmove_count(fen2)} halfmoves")

    # After 1. e4 e5
    fen3 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    assert get_halfmove_count(fen3) == 2, "After 1. e4 e5 should be 2 halfmoves"
    print(f"  ✓ After 1. e4 e5: {get_halfmove_count(fen3)} halfmoves")

    # After 1. e4 e5 2. Nf3
    fen4 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    assert get_halfmove_count(fen4) == 3, "After 1. e4 e5 2. Nf3 should be 3 halfmoves"
    print(f"  ✓ After 1. e4 e5 2. Nf3: {get_halfmove_count(fen4)} halfmoves")

    print("✓ All halfmove count tests passed!\n")


def test_should_use_opening_book():
    """Test the logic for when to use opening book"""
    print("Testing opening book usage decision...")

    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen_middle = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 6"

    assert should_use_opening_book(fen_start, max_halfmoves=10), "Should use book for starting position"
    print(f"  ✓ Starting position uses opening book")

    assert not should_use_opening_book(fen_middle, max_halfmoves=10), "Should not use book for middle game"
    print(f"  ✓ Middle game (11+ halfmoves) does not use opening book")

    print("✓ All opening book usage tests passed!\n")


def test_fetch_opening_book():
    """Test fetching opening book moves from Lichess API"""
    print("Testing opening book API fetch...")

    # Starting position - should have common opening moves
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves = fetch_opening_book_moves(fen_start)

    if moves:
        print(f"  ✓ Fetched {len(moves)} opening moves from API")
        print(f"  Top 3 moves:")
        for i, (move, prob) in enumerate(list(moves.items())[:3]):
            print(f"    {i+1}. {move}: {prob:.2%}")
    else:
        print(f"  ⚠ Warning: No moves returned from API (might be a network issue)")

    # Position after 1. e4 - should have responses
    fen_after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    moves_e4 = fetch_opening_book_moves(fen_after_e4)

    if moves_e4:
        print(f"\n  ✓ Fetched {len(moves_e4)} responses to 1. e4")
        print(f"  Top 3 moves:")
        for i, (move, prob) in enumerate(list(moves_e4.items())[:3]):
            print(f"    {i+1}. {move}: {prob:.2%}")
    else:
        print(f"  ⚠ Warning: No moves returned for 1. e4 response")

    print("\n✓ Opening book API fetch test completed!\n")


def test_integration():
    """Test the full integration with fallback logic"""
    print("Testing full integration with fallback...")

    # Mock model predictions
    model_probs = {'e2e4': 0.25, 'd2d4': 0.20, 'g1f3': 0.15, 'c2c4': 0.10}

    # Test opening position (should use opening book)
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    final_probs, used_book = get_opening_book_or_fallback(fen_start, model_probs)

    print(f"  Starting position:")
    print(f"    Used opening book: {used_book}")
    if used_book:
        print(f"    Top moves from opening book:")
        for i, (move, prob) in enumerate(list(final_probs.items())[:3]):
            print(f"      {i+1}. {move}: {prob:.2%}")
    else:
        print(f"    Fallback to model predictions (API might be unavailable)")

    # Test middle game position (should use model)
    fen_middle = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 6"
    final_probs_mid, used_book_mid = get_opening_book_or_fallback(fen_middle, model_probs)

    print(f"\n  Middle game position:")
    print(f"    Used opening book: {used_book_mid}")
    print(f"    Should use model predictions: {not used_book_mid}")

    print("\n✓ Integration test completed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Maia2 Opening Book Integration")
    print("=" * 60 + "\n")

    test_halfmove_count()
    test_should_use_opening_book()
    test_fetch_opening_book()
    test_integration()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)