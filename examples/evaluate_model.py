"""
Example script for evaluating a trained EmoTiny model.
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from emotiny import load_model, classify_emotion, classify_emotion_with_confidence
from emotiny.preprocessing import EmoTinyPreprocessor
from emotiny.training import EmoTinyTrainer


def benchmark_inference(model_path: str, test_texts: list, num_iterations: int = 100):
    """Benchmark inference speed."""
    print(f"🚀 Benchmarking inference speed ({num_iterations} iterations)...")
    load_model(model_path, use_onnx=True)
    for _ in range(10):
        classify_emotion(test_texts[0]) # Warmup
    start_time = time.time()
    for _ in range(num_iterations):
        for text in test_texts:
            classify_emotion(text)
    end_time = time.time()
    total_inferences = num_iterations * len(test_texts)
    total_time = end_time - start_time
    avg_time = total_time / total_inferences
    throughput = total_inferences / total_time
    print(f"⚡ Benchmark results:")
    print(f"  Total inferences: {total_inferences}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per inference: {avg_time * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} inferences/second")
    return {
        "avg_time_ms": avg_time * 1000,
        "throughput": throughput
    }


def test_multilingual_support():
    """Test multilingual emotion classification."""
    print("🌍 Testing multilingual support...")
    test_cases = [
        # English
        ("I'm so happy today!", "happy"),
        ("This is terrible!", "angry"),
        ("I love you", "love"),
        
        # Portuguese
        ("Estou muito feliz hoje!", "happy"),
        ("Isso é terrível!", "angry"),
        ("Eu te amo", "love"),
        ("Que surpresa incrível!", "surprised"),
        
        # Spanish
        ("¡Estoy muy feliz hoy!", "happy"),
        ("¡Esto es terrible!", "angry"),
        ("Te amo", "love"),
        ("¡Qué sorpresa increíble!", "surprised"),
    ]
    correct_predictions = 0
    total_predictions = len(test_cases)
    for text, expected in test_cases:
        result = classify_emotion_with_confidence(text)
        predicted = result["emotion"]
        confidence = result["confidence"]
        is_correct = predicted == expected
        if is_correct:
            correct_predictions += 1
        status = "✅" if is_correct else "❌"
        print(f"{status} '{text}' -> {predicted} ({confidence:.2f}) [expected: {expected}]")
    accuracy = correct_predictions / total_predictions
    print(f"\n🎯 Multilingual test accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})")
    return accuracy


def test_asr_robustness():
    """Test robustness to ASR-like noise."""
    print("🎤 Testing ASR robustness...")
    # Simulate ASR artifacts
    test_cases = [
        # Original vs ASR-corrupted
        ("I'm so happy today!", "im so happy today"),  # Missing apostrophe
        ("What a surprise!", "what a surprise"),  # Missing punctuation
        ("This is disgusting.", "this is disgusting"),  # Case changes
        ("I love you so much!", "i love you so much"),  # Multiple issues
        ("That's mischievous!", "thats mischievous"),  # Apostrophe issues
    ]
    consistent_predictions = 0
    total_tests = len(test_cases)
    for original, corrupted in test_cases:
        original_emotion = classify_emotion(original)
        corrupted_emotion = classify_emotion(corrupted)
        is_consistent = original_emotion == corrupted_emotion
        if is_consistent:
            consistent_predictions += 1
        status = "✅" if is_consistent else "❌"
        print(f"{status} '{original}' vs '{corrupted}' -> {original_emotion} vs {corrupted_emotion}")
    consistency = consistent_predictions / total_tests
    print(f"\n🎯 ASR robustness: {consistency:.2f} ({consistent_predictions}/{total_tests})")
    return consistency


def main():
    parser = argparse.ArgumentParser(description="Evaluate EmoTiny emotion classifier")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--test-data", help="Path to test CSV dataset")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    parser.add_argument("--label-column", default="emotion", help="Name of label column")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument("--multilingual", action="store_true", help="Test multilingual support")
    parser.add_argument("--asr-robustness", action="store_true", help="Test ASR robustness")
    args = parser.parse_args()
    print("🔍 EmoTiny Model Evaluation")
    print(f"Model: {args.model}")
    load_model(args.model, use_onnx=True)
    print("✅ Model loaded successfully")
    if args.test_data:
        print(f"\n📊 Evaluating on test dataset: {args.test_data}")
        preprocessor = EmoTinyPreprocessor()
        texts, labels = preprocessor.load_dataset_from_csv(args.test_data, args.text_column, args.label_column)
        trainer = EmoTinyTrainer()
        trainer.load_model(args.model)
        results = trainer.evaluate_model(texts, labels)
        print(f"📈 Test Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Macro F1: {results['classification_report']['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    if args.benchmark:
        test_texts = [
            "I'm so happy today!",
            "This is terrible!",
            "What a surprise!",
            "I love you",
            "This is neutral"
        ]
        benchmark_inference(args.model, test_texts)
    if args.multilingual:
        test_multilingual_support()
    if args.asr_robustness:
        test_asr_robustness()
    print("\n🎮 Interactive test (type 'quit' to exit):")
    while True:
        try:
            text = input("Enter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                result = classify_emotion_with_confidence(text)
                print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
                sorted_emotions = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
                print("Top 3 emotions:")
                for emotion, prob in sorted_emotions:
                    print(f"  {emotion}: {prob:.3f}")
                print()
        except KeyboardInterrupt:
            break
    print("👋 Evaluation completed!")


if __name__ == "__main__":
    main()