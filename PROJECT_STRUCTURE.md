# Project Structure Guide

This document explains the importance of each folder and file in this AI text detection research project.

---

##  Root Files

### [README.md](file:///c:/Users/vince/Downloads/scifair/README.md)
**Status:  KEEP - Essential**
- Project overview and documentation
- Usage instructions
- Key findings and results summary
- Critical for GitHub presentation

### [.gitignore](file:///c:/Users/vince/Downloads/scifair/.gitignore)
**Status:  KEEP - Essential**
- Excludes large files (models, datasets) from Git
- Excludes Python cache and IDE files
- Already well-configured for your project

### [humanization.log](file:///c:/Users/vince/Downloads/scifair/humanization.log)
**Status:  REMOVE - Not needed for GitHub**
- 116KB log file from humanization process
- Development artifact, not needed for repository
- Already excluded by `.gitignore` (*.log pattern)

---

##  `/analysis` - Analysis Scripts (5 files)

**Status:  KEEP - Important**

Contains Python scripts for analyzing model behavior and patterns:

1. **[analyze_humanized_triggers.py](file:///c:/Users/vince/Downloads/scifair/analysis/analyze_humanized_triggers.py)** - Analyzes what triggers AI detection in humanized text
2. **[analyze_words.py](file:///c:/Users/vince/Downloads/scifair/analysis/analyze_words.py)** - Word pattern analysis
3. **[compare_models.py](file:///c:/Users/vince/Downloads/scifair/analysis/compare_models.py)** - Compares different model performances
4. **[examine_samples.py](file:///c:/Users/vince/Downloads/scifair/analysis/examine_samples.py)** - Examines individual samples
5. **[highlight_ai_words.py](file:///c:/Users/vince/Downloads/scifair/analysis/highlight_ai_words.py)** - Highlights AI-indicative words

**Recommendation:** Keep all - demonstrates your analytical approach and research methodology.

---

##  `/athena-source-main` - Original Athena Dataset

**Status:  CONSIDER REMOVING/ARCHIVING**

This appears to be the original Athena AI detector source code/dataset.

### Contents:
- `data/` - Original Athena training datasets (AI and human samples, test data)
- `notebooks/` - Jupyter notebooks (2 files)
- `readme.md` - Documentation for original Athena
- `.gitignore`

**Recommendation:** 
- If this is just reference material you downloaded, consider removing it to clean up your repo
- If you modified/used it, keep it but add attribution in your main README
- The datasets here are large and should be excluded via .gitignore

---

##  `/data` - Datasets

**Status:  PARTIALLY KEEP**

### Main Files:
1. **`dataset.csv`** (1.1GB) -  REMOVE
   - Extremely large dataset
   - Already excluded by .gitignore
   
2. **`Balanced_AI_Human_Humanized_UPDATED.csv`** (1.8MB) -  REMOVE
   - Already excluded by .gitignore
   - Too large for GitHub

### `/data/clean` subfolder (4 files):
1. **`clean_dataset.csv`** (2.1MB) -  Consider removing (too large)
2. **`final_dataset.csv`** (2.2MB) -  Consider removing (too large)
3. **`ai_humanized_samples.csv`** (257KB) -  Could keep (sample data)
4. **`ai_to_humanize_limited.csv`** (117KB) -  Could keep (sample data)

**Recommendation:**
- Remove large CSV files (already in .gitignore)
- Keep smaller sample files or create even smaller example datasets (10-20 samples) for demonstration
- Document where full datasets can be obtained

---

##  `/docs` - Documentation (8 files)

**Status:  KEEP ALL - Essential**

All markdown and text files documenting your research:

1. **COMPLETE_RESULTS.md** - Detailed analysis of all results
2. **RESULTS_SUMMARY.md** - Summary of results
3. **detection_discrepancy_analysis.md** - Analysis of detection differences
4. **highlighted_samples.md** - Examples with highlighted patterns
5. **humanized_samples.txt** - Sample humanized texts
6. **trigger_analysis.txt** - Analysis of detection triggers
7. **triggers.txt** - List of trigger words/patterns
8. **word_patterns.txt** - Word pattern documentation

**Recommendation:** Keep all - these demonstrate your research process and findings.

---

##  `/models` - Trained Models

**Status:  ALREADY EXCLUDED - Good**

Contains 20+ model files and 6 subdirectories with trained models:

- Various `.pkl` files (pickle files for scikit-learn models)
- Transformer model directories
- Different model variants (baseline, improved, hybrid, etc.)

**Current Status:**
- Already properly excluded by `.gitignore`
- Models are too large for GitHub (40KB to several MB each)

**Recommendation:**
- Keep excluded from Git
- Document in README how to train/obtain models
- Consider uploading to Google Drive or Hugging Face if you want to share them
- Keep the folder structure in your local project

---

##  `/results` - Results JSON Files (12 files)

**Status:  KEEP - Clean and organized**

Contains JSON files with model evaluation results:

- `athena_athena_data_results.json`
- `athena_improved_results.json`
- `athena_user_humanized_results.json`
- `baseline_results.json`
- `improved_results.json`
- `hybrid_results.json`
- `perplexity_enhanced_full_results.json`
- `perplexity_enhanced_pilot_results.json`
- `perplexity_pilot_results.json`
- `retrained_results.json`
- `structure_results.json`
- `transformer_quick_results.json`

**Cleaned up:** Removed redundant .txt files (comparison.txt, results.txt, comparison_summary.txt)

---

##  `/scripts` - Organized Training and Testing Scripts (22 files)

**Status:  KEEP ALL - Clean and organized**

**Now organized into logical subfolders:**

### 📁 `/scripts/training/` (11 files)

Model training and detector implementation scripts:

1. **athena_train.py** - Main Athena training
2. **athena_train_improved.py** - Improved version
3. **athena_train_user_humanized.py** - Specialized for Undetectable.ai
4. **baseline_detector.py** - Baseline TF-IDF detector
5. **improved_detector.py** - Improved detector
6. **hybrid_detector.py** - Hybrid approach
7. **structure_detector.py** - Structure-based detector
8. **perplexity_detector.py** - Perplexity-based detector
9. **perplexity_enhanced.py** - Enhanced perplexity
10. **transformer_detector.py** - Transformer-based detector
11. **retrain_detectors.py** - Retraining script

### 📁 `/scripts/testing/` (10 files)

Model testing and evaluation scripts:

1. **test_athena.py** - Test Athena baseline
2. **test_athena_improved.py** - Test improved version
3. **test_athena_threshold.py** - Test with different thresholds
4. **test_athena_user.py** - Test user-specific model
5. **test_detector.py** - General testing
6. **test_hybrid.py** - Test hybrid model
7. **test_perplexity.py** - Test perplexity model
8. **test_perplexity_enhanced.py** - Test enhanced perplexity
9. **test_structure.py** - Test structure detector
10. **test_transformer.py** - Test transformer model

### 📁 `/scripts/analysis/` (1 file)

1. **analyze_humanized_quality.py** - Analyzes humanization quality

**Recommendation:** Keep all - scripts are well-organized and demonstrate your complete methodology.

**Note:** All scripts run from project root, so import paths remain unchanged.

---

##  `/util` - Utility Scripts (5 files)

**Status:  KEEP - Supporting code**

Utility functions and helpers:

1. **humanize_samples.py** - Script to humanize AI text samples
2. **load_datasets.py** - Dataset loading utilities
3. **load_humanized_samples.py** - Load humanized samples
4. **merge_datasets.py** - Merge different datasets
5. **word_count.py** - Word counting utility

**Recommendation:** Keep all - these are helper scripts that support your main code.

---

##  Cleanup Recommendations Summary

###  DEFINITELY KEEP:
- `README.md`
- `.gitignore`
- `/analysis/` (all 5 scripts)
- `/docs/` (all 8 documentation files)
- `/results/` (12 JSON files, cleaned up)
- `/scripts/` (22 Python scripts, organized into subfolders)
- `/util/` (all 5 utilities)

###  DEFINITELY REMOVE:
- `humanization.log` (already gitignored)
- Large dataset files in `/data/` (already gitignored)
- All `/models/` files (already gitignored)
- `/scripts/humanized_analysis.txt` (development artifact)

###  CONSIDER:
- **`/athena-source-main/`** - If this is just downloaded source code, remove it. If you modified it, keep and document.
- **Small CSV samples** - Keep 1-2 small sample files (< 500KB) for demonstration, or create even smaller examples

###  TODO BEFORE GITHUB:
1. Run the cleanup script: `python scripts/cleanup_for_github.py`
2. Add a `requirements.txt` or `environment.yml` for dependencies
3. Add a LICENSE file
4. Verify `.gitignore` is working (run `git status`)
5. Consider adding example output/screenshots to `/docs/`
6. Update README with:
   - Installation instructions
   - Dependencies
   - How to obtain datasets
   - How to train models
   - Citation/attribution for Athena if used

---

##  File Size Summary

**Total Project Structure:**
- 8 directories
- ~60+ files
- After cleanup: Most files are small Python scripts and markdown docs
- Large files (datasets, models) already excluded via .gitignore

**GitHub-Ready Size:** < 5MB (after excluding large files)
