import os
import sys
import csv
import cv2
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import HandDetector

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "asl_landmarks.csv")
word_to_record = None

def _init_csv(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        header = ["label"] + [f"x{i}" if j == 0 else f"y{i}" for i in range(42) for j in range(2)]
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def _append_sample(path: str, label: str, features) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([label] + features.tolist())

def terminal_input_thread():
    global word_to_record
    while True:
        try:
            w = input("\nType word to train (or 'q' to quit) and press Enter: ").strip().upper()
            if w == 'Q':
                word_to_record = 'QUIT'
                break
            elif w:
                word_to_record = w
                time.sleep(4)
        except EOFError:
            pass 

def main():
    global word_to_record
    _init_csv(OUTPUT_PATH)
    
    detector = HandDetector(max_hands=2, detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    
    print("\n-- 2-Hand Word Data Collector --")
    print("The camera is opening now. You can see yourself on screen.")
    print("When you are ready, type the word here in the terminal and press Enter!")
    
    threading.Thread(target=terminal_input_thread, daemon=True).start()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        if word_to_record == 'QUIT':
            break
            
        elif word_to_record:
            word = word_to_record
            word_to_record = None 

            for i in range(3, 0, -1):
                start = time.time()
                while time.time() - start < 1.0:
                    ret, cframe = cap.read()
                    if not ret: break
                    cframe = cv2.flip(cframe, 1)
                    cv2.putText(cframe, f"Starting in {i}...", (150, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4)
                    cv2.imshow("Data Collector", cframe)
                    cv2.waitKey(1)

            print(f"RECORDING 100 FRAMES FOR '{word}' - Hold the sign!")
            frames_recorded = 0
            
            while frames_recorded < 100:
                ret, rframe = cap.read()
                if not ret: break
                
                rframe = cv2.flip(rframe, 1)
                annotated = detector.process(rframe)
                features = detector.get_features()
                
                cv2.putText(annotated, f"Recording '{word}': {frames_recorded}/100", (12, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (63, 185, 80), 2, cv2.LINE_AA)
                cv2.imshow("Data Collector", annotated)
                cv2.waitKey(1)
                
                if features is not None:
                    _append_sample(OUTPUT_PATH, word, features)
                    frames_recorded += 1
                    
            print(f"✅ Saved 100 frames for '{word}'.")

        else:
            
            annotated = detector.process(frame)
            cv2.putText(annotated, "Type word in terminal to record...", (12, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Data Collector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

if __name__ == "__main__":
    main()
