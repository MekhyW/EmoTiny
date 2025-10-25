import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from emotiny import EmoTinyTrainer, EmoTinyPreprocessor
from emotiny.optimization import EmoTinyOptimizer


def main():
    parser = argparse.ArgumentParser(description="Train EmoTiny emotion classifier")
    parser.add_argument("--data", default="data/emotions.csv", help="Path to CSV dataset")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    parser.add_argument("--label-column", default="emotion", help="Name of label column")
    parser.add_argument("--output", default="output", help="Output directory for trained model")
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="mlp", help="Classifier type")
    parser.add_argument("--export-onnx", action="store_true", help="Export model to ONNX format")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device for training")
    args = parser.parse_args()
    print("🚀 Starting EmoTiny training pipeline...")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output}")
    print(f"Classifier: {args.classifier}")
    preprocessor = EmoTinyPreprocessor(device=args.device)
    texts, labels = preprocessor.load_dataset_from_csv(args.data, args.text_column, args.label_column)
    trainer = EmoTinyTrainer({"classifier_type": args.classifier, "random_state": 42})
    print("🎯 Training classifier...")
    training_results = trainer.train(texts, labels, save_path=args.output)
    print(f"✅ Training completed!")
    print(f"Test accuracy: {training_results['test_accuracy']:.4f}")
    trainer.plot_confusion_matrix(save_path=os.path.join(args.output, "confusion_matrix.png"))
    if args.export_onnx:
        print("📦 Exporting to ONNX...")
        optimizer = EmoTinyOptimizer()
        embedding_dim = preprocessor.get_embedding_dim()
        onnx_path = os.path.join(args.output, "classifier.onnx")
        optimizer.export_sklearn_to_onnx(trainer.classifier, embedding_dim, onnx_path)
        quantized_path = os.path.join(args.output, "classifier_quantized.onnx")
        optimizer.quantize_onnx_model(onnx_path, quantized_path)
        print("⚡ Benchmarking models...")
        original_results = optimizer.benchmark_model(onnx_path, embedding_dim)
        quantized_results = optimizer.benchmark_model(quantized_path, embedding_dim)
        print(f"Original ONNX: {original_results['average_time_ms']:.2f} ms")
        print(f"Quantized ONNX: {quantized_results['average_time_ms']:.2f} ms")
        print(f"Quantization speedup: {original_results['average_time_ms'] / quantized_results['average_time_ms']}x")
    print(f"🎉 Model saved to: {args.output}")


if __name__ == "__main__":
    main()