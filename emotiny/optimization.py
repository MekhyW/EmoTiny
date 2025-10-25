import os
import numpy as np
import onnxruntime as ort
from typing import Optional, Dict
import joblib
from sklearn.base import BaseEstimator
import warnings
from .config import QUANTIZATION_CONFIG, EMOTION_LABELS


class EmoTinyOptimizer:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the optimizer"""
        self.config = {**QUANTIZATION_CONFIG, **(config or {})}
        
    def export_sklearn_to_onnx(self, classifier: BaseEstimator, input_dim: int, output_path: str, model_name: str = "emotiny_classifier") -> str:
        """Export scikit-learn classifier to ONNX format"""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError("skl2onnx is required for ONNX export. Install with: pip install skl2onnx")
        print(f"Exporting {type(classifier).__name__} to ONNX...")
        initial_type = [('float_input', FloatTensorType([None, input_dim]))]
        onnx_model = convert_sklearn(classifier, initial_types=initial_type, target_opset=self.config["onnx_opset_version"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model exported to: {output_path}")
        self._verify_onnx_model(output_path, input_dim)
        return output_path
    
    def quantize_onnx_model(self, onnx_model_path: str, quantized_model_path: str, quantization_mode: str = "dynamic") -> str:
        """Quantize ONNX model for faster inference"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            warnings.warn("ONNX quantization not available. Skipping quantization.")
            return onnx_model_path
        print(f"Quantizing ONNX model ({quantization_mode})...")
        if quantization_mode == "dynamic":
            quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)
        else:
            warnings.warn("Static quantization not implemented. Using dynamic quantization.")
            quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)
        print(f"Quantized model saved to: {quantized_model_path}")
        original_size = os.path.getsize(onnx_model_path) / 1024 / 1024  # MB
        quantized_size = os.path.getsize(quantized_model_path) / 1024 / 1024  # MB
        compression_ratio = original_size / quantized_size
        print(f"Model size reduction: {original_size:.2f}MB -> {quantized_size:.2f}MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        return quantized_model_path
    
    def optimize_sentence_transformer(self, model_name: str, output_dir: str, optimize_for_cpu: bool = True) -> str:
        """Optimize sentence transformer model for inference"""
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError:
            warnings.warn("Optimum not available. Skipping sentence transformer optimization.")
            return model_name
        print(f"Optimizing sentence transformer: {model_name}")
        model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True, provider="CPUExecutionProvider" if optimize_for_cpu else "CUDAExecutionProvider")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        print(f"Optimized model saved to: {output_dir}")
        return output_dir
    
    def _verify_onnx_model(self, onnx_model_path: str, input_dim: int):
        """Verify ONNX model can be loaded and run."""
        try:
            ort_session = ort.InferenceSession(onnx_model_path)
            dummy_input = np.random.randn(1, input_dim).astype(np.float32)
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: dummy_input})
            print(f"ONNX model verification successful!")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {outputs[0].shape}")
            print(f"Number of classes: {outputs[0].shape[1]}")
            if outputs[0].shape[1] != len(EMOTION_LABELS):
                warnings.warn(f"Output dimension mismatch: expected {len(EMOTION_LABELS)}, got {outputs[0].shape[1]}")
        except Exception as e:
            raise RuntimeError(f"ONNX model verification failed: {e}")
    
    def benchmark_model(self, model_path: str, input_dim: int, num_iterations: int = 1000) -> Dict[str, float]:
        """Benchmark ONNX model inference speed"""
        import time
        print(f"Benchmarking model: {model_path}")
        ort_session = ort.InferenceSession(model_path)
        input_name = ort_session.get_inputs()[0].name
        test_inputs = [np.random.randn(1, input_dim).astype(np.float32) for _ in range(num_iterations)]
        for _ in range(10):
            ort_session.run(None, {input_name: test_inputs[0]}) # Warmup
        start_time = time.time()
        for test_input in test_inputs:
            ort_session.run(None, {input_name: test_input})
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        print(f"Benchmark results:")
        print(f"  Average inference time: {avg_time * 1000:.2f} ms")
        print(f"  Throughput: {throughput:.1f} inferences/second")
        return {"total_time_seconds": total_time, "average_time_ms": avg_time * 1000, "throughput_inferences_per_second": throughput, "num_iterations": num_iterations}
    
    def create_deployment_package(self, classifier_path: str, embedding_model_path: str, output_dir: str, include_onnx: bool = True) -> str:
        """Create a complete deployment package"""
        print("Creating deployment package...")
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        if os.path.isfile(classifier_path):
            shutil.copy2(classifier_path, os.path.join(output_dir, "classifier.joblib"))
        else:
            shutil.copytree(classifier_path, os.path.join(output_dir, "classifier"))
        if include_onnx:
            optimized_embedding_path = os.path.join(output_dir, "embedding_model_optimized")
            self.optimize_sentence_transformer(embedding_model_path, optimized_embedding_path)
        else:
            if os.path.isdir(embedding_model_path):
                shutil.copytree(embedding_model_path, os.path.join(output_dir, "embedding_model"))
        config_path = os.path.join(output_dir, "deployment_config.joblib")
        joblib.dump({"emotion_labels": EMOTION_LABELS, "use_onnx": include_onnx, "optimization_config": self.config}, config_path)
        print(f"Deployment package created at: {output_dir}")
        return output_dir