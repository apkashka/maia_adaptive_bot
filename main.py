from maia2 import model, inference
import math
import chess
import numpy as np


help(inference.inference_each)
# 1) загрузка модели
m = model.from_pretrained(type="rapid", device="gpu")
prepared = inference.prepare()

# 2) ходы игрока (белые), UCI
moves = [
    "d2d4","g1f3","f3d4","c2c3","b1d2","d2f3","c3d4",
    "c1d2","d1d2","e2e3","f1c4","c4d3","d3f5","e1g1","f3e5",
]

elo_oppo = 1100

black_moves = [
    "e7e5", "e5d4", "f8c5", "d8f6", "b8c6",
    "c6d4", "c5b4", "b4d2", "g8e7", "e8g8",
    "d7d5", "c8f5", "e7f5", "f5h4", "f6g5",
]

ELOS = [700, 850, 1000, 1150, 1300, 1450, 1600, 1750, 1900]
ce = {elo: 0.0 for elo in ELOS}

board = chess.Board()
probs = {elo: [] for elo in ELOS}
max_ps = {elo: [] for elo in ELOS}   # ← НОВОЕ


# ===== main loop =====
for wm, bm in zip(moves, black_moves):
    fen = board.fen()

    for elo in ELOS:
        p_dict, _ = inference.inference_each(
            m, prepared, fen,
            elo_self=elo, elo_oppo=elo_oppo
        )

        probs[elo].append(p_dict.get(wm, 1e-9))
        max_ps[elo].append(max(p_dict.values()))   # ← НОВОЕ

    board.push_uci(wm)
    board.push_uci(bm)

# ===== вывод =====
print("\n=== Naive average probability ===")
for elo in ELOS:
    print(f"Elo {elo}: avgP = {np.mean(probs[elo]):.4f}")

print("\n=== Cross-Entropy (CE) ===")
for elo in ELOS:
    ce = -sum(math.log(p) for p in probs[elo]) / len(probs[elo])
    print(f"Elo {elo}: CE = {ce:.4f}")

print("\n=== max_P (position difficulty) ===")
for elo in ELOS:
    print(f"Elo {elo}: max_P = {np.mean(max_ps[elo]):.4f}")