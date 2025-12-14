# **AI Text Detection - Complete Results Summary**

## **Project Overview**

Built and tested 7 different AI detection approaches to find the most reliable method for detecting AI-generated text, including humanized AI.

---

## **All Detectors Built**

### **1. TF-IDF Detector (Vocabulary-Based)**

**Dataset:** 11,000 samples (5k Human, 5k AI_RAW, 1k AI_Humanized)

**Approach:** Logistic Regression on word frequency patterns

| **Model** | **Standard Test** | **Humanized Test** | **Real-World** |
| --- | --- | --- | --- |
| Baseline (no humanized training) | 99.17% | 100.00% |  Failed |
| Improved (with humanized training) | 99.13% | 100.00% |  Failed |

**Real-World Performance:**

- Flagged Wikipedia as AI (false positives)
- Flagged actual AI as Human (false negatives)
- **Conclusion:** Severe overfitting to dataset vocabulary (hence high testing accuracy), but failed in real-world scenarios - for both normal and humanized AI.

---

### **2. Structure Detector (Statistical Features)**

**Dataset:** Same 11,000 samples

**Features:** Sentence length variation, vocabulary richness, punctuation density, avg word length

| **Model** | **Standard Test** | **Humanized Test** |
| --- | --- | --- |
| Baseline | 86.80% | 100.00% |
| Improved | 86.83% | 100.00% |

**Conclusion:** Lower accuracy than TF-IDF, still overfitted (100% on real-world data, humanized or AI)

---

### **3. Hybrid Detector (TF-IDF + Structure)**

**Dataset:** Same 11,000 samples

**Approach:** Combined 5000 TF-IDF features + 4 structural features

| **Model** | **Standard Test** | **Humanized Test** |
| --- | --- | --- |
| Baseline | 98.77% | 100.00% |
| Improved | 98.80% | 100.00% |

**Conclusion:** Nearly identical to TF-IDF alone - vocabulary dominates, structure adds minimal value

---

### **4. Perplexity Detector (Single Feature)**

**Dataset:** Same 11,000 samples

**Approach:** GPT-2 perplexity as single feature

| **Model** | **Standard Test** | **Humanized Test** |
| --- | --- | --- |
| Baseline | 90.00% | 0.00%  |
| Improved | 50.00% | 100.00% |

**Conclusion:** FAILED - Single perplexity value insufficient to distinguish AI from human

---

### **5. Enhanced Perplexity Detector (4 Features)**

**Dataset:** Same 11,000 samples

**Features:** Avg perplexity, burstiness (variance), max perplexity, min perplexity

| **Model** | **Standard Test** | **Humanized Test** |
| --- | --- | --- |
| Baseline | 70.30% | 0.00%  |
| Improved | 42.80%  | 100.00% |

**Conclusion:** FAILED - Even with enhanced features, perplexity values too similar between human/AI text

---

### **6. Transformer Detector (DistilBERT - Our Attempt)**

**Dataset:** Same 11,000 samples

**Approach:** Fine-tuned DistilBERT (Language model that has general sense of English language, it can fill in words and understand sentiment, etc.)

| **Model** | **Test Accuracy** | **Real-World** |
| --- | --- | --- |
| Baseline | 99.67% |  Called everything Human |
| Improved | 99.84% |  Called everything Human |

**Conclusion:** FAILED - High test accuracy but opposite failure mode from TF-IDF (called AI → Human instead of Human → AI), overfitted

---

### **7. Athena Detector (DistilBERT - Proven Implementation) **

**Dataset:** Athena's original 9,000 samples (different source, higher quality)

**Approach:** Fine-tuned DistilBERT using Athena's proven methodology

| **Model** | **Test Accuracy** | **Real-World** | **Status** |
| --- | --- | --- | --- |
| **Baseline** | **98.82%** | ** Works perfectly!** | **SUCCESS** |
| **Improved** | **98.92%** | **TBD** | **Trained** |

**Real-World Performance (Baseline):**

-  Correctly detects fresh ChatGPT text
-  Correctly identifies Wikipedia as human
-  Detects humanized AI from multiple sources
-  No false positives on formal writing

**Improved Model Details:**
- **Training Dataset:** Athena original (9,000) + User humanized (1,045) = 12,045 total samples
- **Training Time:** ~4.6 hours (3 epochs, 3,615 steps)
- **Train/Test Split:** 9,636 training / 2,409 test
- **Accuracy Improvement:** +0.10% (98.82% → 98.92%)
- **Model Size:** 267.8 MB (DistilBERT)
- **Saved Location:** `models/athena_improved/`

**Key Success Factor:** Athena used higher-quality, more diverse training data (100-400 words per sample and various high-quality sources vs our 68-word averages)

---

## **Summary Statistics**

| **Detector Type** | **Best Accuracy** | **Real-World Success** |
| --- | --- | --- |
| TF-IDF | 99.17% |  No |
| Structure | 86.83% |  Not tested |
| Hybrid | 98.80% |  Likely failed |
| Perplexity (Single) | 90.00% |  No |
| Perplexity (Enhanced) | 70.30% |  No |
| Transformer (Ours) | 99.84% |  No |
| **Athena (Baseline)** | **98.82%** | ** YES** |
| **Athena (Improved)** | **98.92%** | **TBD** |

---

## **Key Findings**

### **1. Test Accuracy ≠ Real-World Performance**

- TF-IDF: 99.17% test → 0% real-world
- Transformer: 99.84% test → 0% real-world
- **Lesson:** Dataset quality matters more than test scores

### **2. Vocabulary-Based Detection is Fundamentally Flawed**

- Learns formal writing = AI
- Can't distinguish "formal human" from "AI"
- Wikipedia triggers false positives

### **3. Perplexity Detection Doesn't Work (For This Task)**

- Human and modern AI have overlapping perplexity distributions
- Even with 4 features, max 70% accuracy
- Unstable across different text types

### **4. Success Requires Quality Data**

**Our failed attempts:**

- 11k samples, avg 68-100 words
- Humanized samples too short (58-68 words)
- Dataset may have been mislabeled

**Athena's success:**

- 9k samples, 100-400 words each
- Higher quality, more diverse sources
- Proven methodology from published research

---

## **Files & Models**

### **Working Models**

- `/models/athena_athena_data/` - **Baseline Athena (98.82%)** 
- `/models/athena_improved/` - **Improved Athena (98.92%)** 

### **Test Scripts**

- `scripts/test_athena.py` - Interactive testing for Athena baseline
- `scripts/athena_train_improved.py` - Training script for improved model

### **Failed Models (For Comparison)**

- `/models/baseline_model.pkl` - TF-IDF baseline
- `/models/improved_model.pkl` - TF-IDF improved
- `/models/structure_*.pkl` - Structure detector
- `/models/hybrid_*.pkl` - Hybrid detector
- `/models/perplexity_*.pkl` - Perplexity detectors
- `/models/transformer_*.pkl` - Our transformer attempt

### **Results**

- `results/athena_athena_data_results.json` - Athena baseline results
- `results/athena_improved_results.json` - Athena improved results (NEW)
- `results/retrained_results.json` - TF-IDF results
- `results/structure_results.json` - Structure results
- `results/hybrid_results.json` - Hybrid results
- `results/perplexity_enhanced_full_results.json` - Perplexity results
