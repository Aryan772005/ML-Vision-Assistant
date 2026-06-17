import os
import math
import random
import numpy as np
import csv

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "asl_landmarks.csv")
SAMPLES_PER_CLASS = 300
NOISE_STD = 0.025   

def _base_hand() -> dict:

    return {
        0:  (0.00,  0.00),   
        1:  (-0.30,  0.25),  
        2:  (-0.45,  0.45),  
        3:  (-0.55,  0.60),  
        4:  (-0.60,  0.75),  
        5:  (-0.15,  0.55),  
        6:  (-0.15,  0.75),  
        7:  (-0.15,  0.90),  
        8:  (-0.15,  1.00),  
        9:  (0.00,   0.55),  
        10: (0.00,   0.75),  
        11: (0.00,   0.92),  
        12: (0.00,   1.00),  
        13: (0.15,   0.55),  
        14: (0.15,   0.72),  
        15: (0.15,   0.88),  
        16: (0.15,   0.98),  
        17: (0.28,   0.50),  
        18: (0.28,   0.65),  
        19: (0.28,   0.78),  
        20: (0.28,   0.88),  
    }

def _closed_fist() -> dict:

    return {
        0:  (0.00,  0.00),
        1:  (-0.25,  0.20),
        2:  (-0.35,  0.35),
        3:  (-0.40,  0.45),
        4:  (-0.35,  0.50),
        5:  (-0.15,  0.40),
        6:  (-0.10,  0.55),
        7:  (-0.08,  0.60),
        8:  (-0.05,  0.62),
        9:  (0.00,   0.40),
        10: (0.05,   0.55),
        11: (0.05,   0.60),
        12: (0.05,   0.62),
        13: (0.15,   0.40),
        14: (0.18,   0.52),
        15: (0.18,   0.58),
        16: (0.18,   0.60),
        17: (0.28,   0.38),
        18: (0.30,   0.48),
        19: (0.30,   0.54),
        20: (0.30,   0.56),
    }

def _pose_for_letter(letter: str) -> dict:
    base  = _base_hand()
    fist  = _closed_fist()

    poses = {
        
        "A": {**fist, 4: (-0.28, 0.42)},  
        "E": {**fist, 4: (-0.20, 0.55), 8: (-0.08, 0.55), 12: (0.05, 0.55), 16: (0.18, 0.55), 20: (0.30, 0.52)},
        "M": {**fist, 8: (-0.08, 0.50), 12: (0.05, 0.50), 16: (0.18, 0.50), 4: (-0.10, 0.52)},
        "N": {**fist, 8: (-0.08, 0.50), 12: (0.05, 0.50), 4: (-0.10, 0.52)},
        "S": {**fist, 4: (-0.20, 0.52)},

        "D": {**fist, 8: (-0.15, 1.00), 4: (-0.42, 0.82)},   
        "G": {**fist, 8: (-0.15, 0.60), 7: (-0.15, 0.50), 6: (-0.15, 0.45), 5: (-0.15, 0.42)},  
        "H": {**fist, 8: (-0.15, 0.60), 12: (0.00, 0.60), 6: (-0.15, 0.48), 10: (0.00, 0.48)},  
        "X": {**fist, 5: (-0.15, 0.42), 6: (-0.15, 0.52), 7: (-0.10, 0.62), 8: (-0.05, 0.68)},  
        "Z": {**fist, 8: (-0.15, 0.90), 7: (-0.05, 0.75), 6: (-0.15, 0.55), 5: (-0.15, 0.42)},  

        "B": {**base, 4: (-0.30, 0.45), 1: (-0.28, 0.40), 2: (-0.32, 0.38), 3: (-0.34, 0.40)},  
        "C": {**base, 4: (-0.52, 0.70), 8: (-0.25, 0.90), 12: (-0.05, 0.92), 16: (0.10, 0.90), 20: (0.25, 0.82)},
        "F": {**base, 8: (-0.40, 0.60), 4: (-0.38, 0.62)},  
        "I": {**fist, 20: (0.28, 0.88)},   
        "J": {**fist, 20: (0.28, 0.88), 19: (0.32, 0.72), 18: (0.38, 0.58)},  
        "K": {**fist, 8: (-0.15, 0.90), 12: (-0.02, 0.80), 4: (-0.30, 0.75)},
        "L": {**fist, 8: (-0.15, 1.00), 7: (-0.15, 0.85), 6: (-0.15, 0.70), 5: (-0.15, 0.55), 4: (-0.58, 0.75)},
        "O": {**base, 4: (-0.35, 0.82), 8: (-0.28, 0.88), 12: (-0.08, 0.90), 16: (0.08, 0.88), 20: (0.20, 0.82)},
        "P": {**fist, 8: (-0.05, 0.60), 12: (0.05, 0.62), 4: (-0.30, 0.68)},
        "Q": {**fist, 8: (-0.10, 0.62), 4: (-0.32, 0.72)},  
        "R": {**fist, 8: (-0.15, 0.90), 12: (-0.05, 0.85)},  
        "T": {**fist, 4: (-0.20, 0.52), 8: (-0.05, 0.55)},
        "U": {**fist, 8: (-0.15, 0.90), 12: (0.00, 0.90)},   
        "V": {**fist, 8: (-0.20, 0.95), 12: (0.05, 0.95)},   
        "W": {**fist, 8: (-0.22, 0.92), 12: (0.00, 0.95), 16: (0.18, 0.90)},  
        "Y": {**fist, 4: (-0.58, 0.75), 20: (0.28, 0.88)},   
        "SPACE":     {**base},                                  
        "BACKSPACE": {**fist, 4: (-0.55, 0.62)},               
        "NOTHING":   {**fist},                                  
    }

    if letter not in poses:
        poses[letter] = _base_hand()

    return poses[letter]

def pose_to_feature(pose: dict, noise_std: float = 0.0) -> np.ndarray:

    coords = np.array([pose.get(i, (0.0, 0.0)) for i in range(21)], dtype=np.float32)

    if noise_std > 0:
        coords += np.random.normal(0, noise_std, coords.shape).astype(np.float32)

    coords -= coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords.flatten()

def generate_dataset(output_path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "BACKSPACE", "NOTHING"]

    header = ["label"] + [f"x{i}" if j == 0 else f"y{i}"
                          for i in range(21) for j in range(2)]

    rows = []
    for label in labels:
        pose = _pose_for_letter(label)
        for _ in range(SAMPLES_PER_CLASS):
            features = pose_to_feature(pose, noise_std=NOISE_STD)
            rows.append([label] + features.tolist())

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    total = len(labels) * SAMPLES_PER_CLASS
    print(f"[generate] Wrote {total} samples ({SAMPLES_PER_CLASS}/class) -> {output_path}")

if __name__ == "__main__":
    generate_dataset()
