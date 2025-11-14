import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from emotiny import load_model, classify_emotion

load_model("output")

while True:
    text = input("Type input text: ")
    emotion = classify_emotion(text)
    print(emotion)
