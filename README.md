# AI Text Detection Research

A comprehensive research project evaluating multiple AI text detection approaches for identifying AI-generated and humanized AI text.

## 🎯 Key Results

- **Athena Baseline:** 98.82% accuracy
- **Athena Improved:** 98.92% accuracy  
- **Athena User Humanized:** 98.90% accuracy (specialized for Undetectable.ai)

## 📊 Models Tested

1. TF-IDF (99.17% test, failed in real-world)
2. Structure Detector (86.83%)
3. Hybrid TF-IDF+Structure (98.80%, failed in real-world)
4. Perplexity Single Feature (90%, failed)
5. Enhanced Perplexity (70.30%, failed)
6. Transformer (99.84% test, failed in real-world)
7. **Athena Baseline** (98.82%, SUCCESS)
8. **Athena Improved** (98.92%, SUCCESS)
9. **Athena User Humanized** (98.90%, SUCCESS)

## 🔍 Key Findings

- Test accuracy does not equal real-world performance
- Dataset quality matters more than model complexity
- Humanizer detection is tool-specific, not universal
- Training on Undetectable.ai samples enables detection of that specific humanizer

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/scifair.git
cd scifair
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support with PyTorch, visit [PyTorch.org](https://pytorch.org/get-started/locally/) for CUDA-specific installation instructions.

## 📂 Project Structure

```
scifair/
├── analysis/              # Analysis scripts for model behavior
├── docs/                  # Documentation and research findings
├── results/               # JSON result files from experiments
├── scripts/
│   ├── training/          # Model training scripts (11 files)
│   │   ├── athena_train*.py
│   │   ├── *_detector.py
│   │   └── retrain_detectors.py
│   ├── testing/           # Model testing scripts (10 files)
│   │   └── test_*.py
│   └── analysis/          # Script analysis tools
└── util/                  # Utility functions
```

## 📖 Usage

### Testing Pre-trained Models

```bash
# Test baseline model
python scripts/testing/test_athena.py

# Test with adjusted threshold (5% instead of 50%)
python scripts/testing/test_athena_threshold.py baseline

# Test Undetectable.ai specialist
python scripts/testing/test_athena_threshold.py user
```

### Training Your Own Models

```bash
# Train Athena baseline
python scripts/training/athena_train.py

# Train improved version
python scripts/training/athena_train_improved.py

# Train specialized humanized detector
python scripts/training/athena_train_user_humanized.py
```

## 📊 Datasets

**Note:** Large model files and datasets are excluded from this repository due to size constraints.

### Required Datasets

You'll need to prepare your own datasets with the following structure:
- Training data: CSV files with `text` and `label` columns
- Label 0: Human-written text
- Label 1: AI-generated text
- Label 2: Humanized AI text (optional, for specialized models)

### Sample Dataset Format

```csv
text,label
"Human written text example",0
"AI generated text example",1
"Humanized AI text example",2
```

## 📚 Documentation

- [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) - Complete results summary
- [`docs/COMPLETE_RESULTS.md`](docs/COMPLETE_RESULTS.md) - Detailed analysis
- [`results/`](results/) - JSON result files from all experiments
- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) - Detailed explanation of project structure

## 🔬 Research Methodology

This project systematically evaluated various approaches to AI text detection:
1. **Traditional ML**: TF-IDF with Logistic Regression
2. **Structural Analysis**: Sentence length, punctuation patterns
3. **Perplexity-based**: Using GPT-2 perplexity scores
4. **Transformer-based**: Fine-tuned DistilBERT models (Athena)

## 🙏 Attribution

This project builds upon the Athena AI detector framework. The original Athena dataset and baseline model provided the foundation for our improvements and specialized variants.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⚠️ Disclaimer

This research is for educational purposes. AI detection is an evolving field, and no detector is 100% accurate. Use these tools responsibly and in conjunction with other verification methods.
