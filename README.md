# 🎭 EmoTiny

**Fast multilingual emotion classification for real-time animated face control**

EmoTiny is a lightweight, optimized emotion classification system designed specifically for real-time applications like animated face control. It provides low-latency emotion detection from short text inputs (1-4 sentences) with support for multiple languages and robustness to ASR (Automatic Speech Recognition) noise.

## Architecture

EmoTiny uses a two-stage approach:
1. **Sentence Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
2. **Classification**: Logistic Regression or MLP classifier

This design provides the optimal balance between accuracy, speed, and model size for real-time applications.

## Quick Start

### Installation

```bash
git clone https://github.com/your-repo/emotiny.git
cd emotiny
pip install -r requirements.txt
```

### Basic Usage

```python
from emotiny import load_model, classify_emotion
load_model("path/to/trained/model")
emotion = classify_emotion("I'm so happy today!")
print(emotion)  # "happy"
```

### Real-time Integration

```python
from emotiny import load_model, classify_emotion

load_model("./models/emotiny")

def update_face_animation(transcribed_text):
    """Update animated face based on speech emotion."""
    emotion = classify_emotion(transcribed_text)
    face_controller.set_emotion(emotion)
    return emotion

while True:
    audio = capture_audio()
    text = whisper_transcribe(audio)
    emotion = update_face_animation(text)
    print(f"Detected emotion: {emotion}")
```

## Training Your Own Model

### 1. Prepare Dataset

Option A: Use the provided generator to build a dataset from raw texts using a local judge LLM via Ollama.

1) Install and start Ollama, then pull the desired model:

```bash
ollama pull gemma3:1b
ollama serve
```

2) Put your raw `.txt` or `.md` files under `./data/` (subfolders allowed).

3) Run the dataset generator:

```bash
python scripts/generate_dataset.py \
  --data-dir ./data \
  --output-parquet ./data/emotions.parquet \
  --output-csv ./data/emotions.csv \
  --model gemma3:1b
```

This will create `text` and `emotion` columns labeled with one of the 7 supported emotions.

Option B: Create a CSV file manually with `text` and `emotion` columns:

```csv
text,emotion
"I'm so happy today!",happy
"This is terrible",angry
"What a surprise!",surprised
"Estou muito feliz",happy
"¡Qué sorpresa!",surprised
```

### 2. Train Model

```bash
python scripts/train_model.py \
    --data ./data/emotions.csv \
    --output ./models/emotiny \
    --classifier mlp \
    --export-onnx
```

### 3. Evaluate Model

```bash
python scripts/evaluate_model.py \
    --model ./models/emotiny \
    --test-data ./data/emotions.csv \
    --benchmark \
    --multilingual \
    --asr-robustness
```

## Supported Emotions

| Emotion | Description | Example Texts |
|---------|-------------|---------------|
| `neutral` | Neutral/calm state | "The meeting will be at 8pm", "Got it, I understand" |
| `happy` | Joy, happiness | "I'm so happy!", "Yay this is great!" |
| `sad` | Sadness, melancholy | "I'm feeling down", "My dog just died..." |
| `angry` | Anger, frustration | "I'm furious!", "Goddamit you had ONE job" |
| `surprised` | Surprise, shock, fear | "What the hell is this??", "I can't believe it!" |
| `disgusted` | Disgust, revulsion | "That's disgusting", "Eww, gross!" |
| `mischievous` | Playful, sassy | "You're being naughty", "Come here often, sweetie?" |

## Advanced Configuration

### Custom Training Configuration

```python
from emotiny import EmoTinyTrainer

config = {
    "classifier_type": "mlp",  # or "logistic"
    "mlp_hidden_sizes": (128, 64),
    "mlp_activation": "relu",
    "test_size": 0.2,
    "cross_validation_folds": 5
}

trainer = EmoTinyTrainer(config)
```

### ONNX Optimization

```python
from emotiny.optimization import EmoTinyOptimizer
optimizer = EmoTinyOptimizer()
optimizer.export_sklearn_to_onnx(classifier, input_dim=384, output_path="model.onnx")
optimizer.quantize_onnx_model("model.onnx", "model_quantized.onnx")
```

## Technical Details

### Embedding Model Choice

We chose `paraphrase-multilingual-MiniLM-L12-v2` because:
- **Multilingual**: Supports 50+ languages
- **Fast**: 384-dimensional embeddings
- **Optimized**: Designed for sentence-level tasks
- **Small**: ~120MB model size

### Classifier Options

**Logistic Regression**:
- Pros: Extremely fast, interpretable
- Cons: Limited capacity for complex patterns
- Best for: Simple datasets, maximum speed

**MLP (Multi-Layer Perceptron)**:
- Pros: Better accuracy, handles non-linear patterns
- Cons: Slightly slower, more parameters
- Best for: Complex datasets, balanced speed/accuracy