"""
Quick start example for EmoTiny emotion classification.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from emotiny import get_supported_emotions


def main():
    print("🎭 EmoTiny Quick Start Example")
    print("=" * 40)
    emotions = get_supported_emotions()
    print(f"Supported emotions: {', '.join(emotions)}")
    print()
    model_path = "path/to/your/trained/model"  # Update this path
    try:
        # Load model (uncomment when you have a trained model)
        # load_model(model_path)
        # print(f"✅ Model loaded from: {model_path}")
        
        # For demonstration, we'll show what the API looks like
        print("📝 Example usage (after loading a trained model):")
        print()
        example_texts = [
            "I'm so happy today!",
            "This is terrible, I hate it.",
            "I love you so much!",
            "What a surprise!",
            "I'm feeling neutral about this.",
            "That's disgusting.",
            "You're being quite mischievous!",
            "This is a nightmare.",
            "I'm really angry about this situation.",
            # Portuguese examples
            "Estou muito feliz hoje!",
            "Eu te amo muito!",
            "Que surpresa incrível!",
            # Spanish examples
            "¡Estoy muy feliz hoy!",
            "¡Te amo mucho!",
            "¡Qué sorpresa increíble!"
        ]
        print("Example API calls:")
        print()
        for text in example_texts[:5]:
            print(f"Text: '{text}'")
            print(f"  classify_emotion(text) -> 'predicted_emotion'")
            print(f"  classify_emotion_with_confidence(text) -> {{")
            print(f"    'emotion': 'predicted_emotion',")
            print(f"    'confidence': 0.85,")
            print(f"    'probabilities': {{'happy': 0.85, 'neutral': 0.10, ...}}")
            print(f"  }}")
            print()
        print("🚀 Real-time usage example:")
        print("""
# For real-time animated face control:
def update_face_animation():
    # Get text from speech recognition (e.g., Whisper)
    transcribed_text = get_speech_transcription()
    
    # Classify emotion
    emotion = classify_emotion(transcribed_text)
    
    # Update animated face
    face_controller.set_emotion(emotion)
    
    # Optional: Get confidence for smoother transitions
    result = classify_emotion_with_confidence(transcribed_text)
    if result['confidence'] > 0.7:
        face_controller.set_emotion(result['emotion'])
    else:
        # Keep previous emotion or use neutral
        face_controller.set_emotion('neutral')
""")
        print("⚡ Performance characteristics:")
        print("  - Inference time: ~1-5ms per text (CPU)")
        print("  - Model size: ~50-100MB (including embeddings)")
        print("  - Memory usage: ~200-500MB")
        print("  - Supports: English, Portuguese, Spanish")
        print("  - Robust to ASR noise and transcription errors")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 To use this example:")
        print("1. Train a model using examples/train_model.py")
        print("2. Update the model_path variable above")
        print("3. Run this script again")
    print("\n📚 Next steps:")
    print("1. Prepare your emotion dataset (CSV with 'text' and 'emotion' columns)")
    print("2. Train a model: python examples/train_model.py --data your_data.csv --output ./models/emotiny")
    print("3. Evaluate: python examples/evaluate_model.py --model ./models/emotiny --benchmark")
    print("4. Integrate into your animated face project!")


if __name__ == "__main__":
    main()