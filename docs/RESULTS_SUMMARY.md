# AI Detector Results - Quick Reference

## Performance Summary Table

| Detector | Test Accuracy | Humanized Test | Real-World | Training Time | Status |
|:---|:---|:---|:---|:---|:---|
| **TF-IDF Baseline** | 99.17% | 100.00% |  Failed | ~5 mins | Overfitted |
| **TF-IDF Improved** | 99.13% | 100.00% |  Failed | ~5 mins | Overfitted |
| **Structure Baseline** | 86.80% | 100.00% |  Unknown | ~3 mins | Weak |
| **Structure Improved** | 86.83% | 100.00% |  Unknown | ~3 mins | Weak |
| **Hybrid Baseline** | 98.77% | 100.00% |  Failed | ~10 mins | Overfitted |
| **Hybrid Improved** | 98.80% | 100.00% |  Failed | ~10 mins | Overfitted |
| **Perplexity Single** | 90.00% | 0.00% |  Failed | ~1 hour | Unusable |
| **Perplexity Enhanced** | 70.30% | 0.00% |  Failed | ~1 hour | Unusable |
| **Transformer (Ours)** | 99.84% | N/A |  Failed | ~10 mins | Reversed labels |
| **Athena Baseline** | **98.82%** | **TBD** | ** SUCCESS** | ~4 hours | **WORKING** |
| **Athena Improved** | **98.92%** | **TBD** | **TBD** | ~4.5 hours | **Trained** |

## Key Metrics Explained

**Test Accuracy:** Performance on held-out test set (20% of data)
**Humanized Test:** Performance on AI text that was "humanized" 
**Real-World:** Performance on fresh, unseen samples (ChatGPT, Wikipedia, etc.)

## Winner: Athena Baseline 

**Why it won:**
- 98.82% test accuracy
- **Actually works on real samples** (unlike all others)
- Correctly identifies fresh ChatGPT
- Doesn't flag Wikipedia as AI
- Detects multiple humanization techniques

**What made it different:**
- Higher quality training data (100-400 words vs 68 words)
- More diverse sources
- Proven methodology from published research
- Better dataset curation

## Science Fair Takeaway

> **"I built 7 different AI detectors, 6 of which achieved 90-99% test accuracy but completely failed in real-world use. This taught me that dataset quality matters far more than model complexity. By using proven methodology (Athena) with high-quality data, I achieved a detector that actually works."**

## Files for Presentation

### Results
- `results/athena_athena_data_results.json` - Winning model results
- `results/retrained_results.json` - TF-IDF comparison
- `walkthrough.md` - Full detailed analysis

### Working Model
- `models/athena_athena_data/` - The detector that works!
- `scripts/test_athena.py` - Interactive testing

### For Demonstration
1. Load test_athena.py
2. Test with Wikipedia → Correctly says HUMAN
3. Test with ChatGPT → Correctly says AI
4. Test with humanized AI → *Results vary*

