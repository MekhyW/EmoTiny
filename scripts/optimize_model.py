import sys
import argparse
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from emotiny import EmoTinyTrainer
from emotiny.optimization import EmoTinyOptimizer

def main():
    parser = argparse.ArgumentParser(description="Optimize EmoTiny ONNX model")
    parser.add_argument("--output", default="output", help="Output directory for optimized model")
    parser.add_argument("--model", default="output", help="Path to trained model directory")
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="mlp", help="Classifier type")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device for optimization")
    args = parser.parse_args()
    print("📦 Exporting to ONNX...")
    trainer = EmoTinyTrainer({"classifier_type": args.classifier, "random_state": 42})
    trainer.load_model(args.model)
    trainer.preprocessor.device = args.device
    optimizer = EmoTinyOptimizer()
    embedding_dim = trainer.preprocessor.get_embedding_dim()
    onnx_path = os.path.join(args.output, "classifier.onnx")
    if trainer.classifier is None:
        raise RuntimeError("No classifier loaded. Train a model first or provide a valid --model path.")
    optimizer.export_sklearn_to_onnx(trainer.classifier, embedding_dim, onnx_path)
    quantized_path = os.path.join(args.output, "classifier_quantized.onnx")
    optimizer.quantize_onnx_model(onnx_path, quantized_path)
    print("⚡ Benchmarking models...")
    original_results = optimizer.benchmark_model(onnx_path, embedding_dim)
    quantized_results = optimizer.benchmark_model(quantized_path, embedding_dim)
    print(f"Original ONNX: {original_results['average_time_ms']:.2f} ms")
    print(f"Quantized ONNX: {quantized_results['average_time_ms']:.2f} ms")
    print(f"Quantization speedup: {original_results['average_time_ms'] / quantized_results['average_time_ms']}x")

if __name__ == "__main__":
    main()