# Clinical Relevance Evaluation

> **Automated validation of generated dermatological concepts using BioBERT semantic similarity with automatic threshold determination**

## Overview

This framework evaluates whether automatically generated dermatological concepts are clinically meaningful by comparing them against established medical vocabularies. It tests **three critical hypotheses** with automatic thresholds adapted to each vocabulary.

## Quick Start

```bash
# 1. Verify setup (recommended)
python eval/test_setup.py

# 2. Run evaluation (~15-30 min)
./eval/run_clinical_relevance.sh

# 3. Generate visualizations (~1-2 min)
python eval/analyze_clinical_relevance.py
```

## What It Does

### Input
- **Generated Concepts**: 18 configurations from your pipeline
- **Reference Vocabularies**:
  - Derm7pt (28 concepts) - Human-annotated dermatology
  - Dermlike (10 concepts) - Literature-based features  
  - ICD-10-CM (28,871 concepts) - General medical codes

### Process
```
Generated Concepts → BioBERT Embeddings → Cosine Similarity → Statistical Analysis
```

### Output
- **18 detailed JSON files** with concept mappings
- **1 summary CSV** with aggregate statistics
- **4 visualization plots** showing results
- **Comparative analysis** across vocabularies

## Three Evaluation Points

### Point 1: Clinical Relevance
**Test**: Generated concepts should be more similar to clinical vocabularies than general medical terms

**Metrics**: Mean similarity to Derm7pt/Dermlike vs ICD-10-CM

**Success**: Clinical > 0.60, General < 0.50, Improvement > 30%

### Point 2: Specificity & Representativeness
**Test**: Generated concepts should map to SOME (not ALL) manual concepts

**Metrics**: 
- Average mappings per concept (target: 2-5)
- Specificity score (target: >0.15)
- Distribution: more few-mappings than many-mappings

**Success**: Concepts are discriminative, not generic

**Key Innovation**: **Automatic threshold determination** based on inter-concept distances within each vocabulary (instead of fixed 0.7)

### Point 3: Mapping Quality
**Test**: Generated→Clinical similarity should be comparable to Clinical↔Clinical baseline

**Metrics**: Compare Generated→Derm7pt/Dermlike vs Derm7pt↔Dermlike

**Success**: Generated concepts align as well as clinical vocabularies align with each other

## Expected Results

**Example:**
```
POINT 1: Clinical Relevance
  DERM7PT    : 0.6842 ± 0.0234  ← Clinically relevant!
  DERMLIKE   : 0.6523 ± 0.0198  ← Domain-specific!
  ICD10CM    : 0.4156 ± 0.0312  ← Properly distinct!
  Improvement: 60.5% ✓ PASS

POINT 2: Specificity
  Avg mappings: 2.3  ✓ (maps to some, not all)
  Specificity score: 0.2145  ✓
  Auto threshold: 0.6765 (adapted to vocabulary)

POINT 3: Mapping Quality
  Generated → Clinical: 0.6683
  Derm7pt ↔ Dermlike:  0.5234
  Ratio: 1.28x ✓ EXCELLENT
```

## File Structure

```
eval/
├── README.md                          # This file
├── QUICKSTART.md                      # Detailed usage guide
├── INDEX.md                           # Complete file reference
├── SUMMARY.md                         # Technical documentation
├── concept_mapping_emb.py             # Main evaluation (BioBERT)
├── concept_mapping_gemini.py          # Alternative (Gemini API)
├── analyze_clinical_relevance.py     # Visualization & stats
├── run_clinical_relevance.sh         # Execution wrapper
├── test_setup.py                      # Setup verification
└── requirements.txt                   # Dependencies
```

## Documentation Guide

| Need | Read |
|------|------|
| Quick start | `QUICKSTART.md` |
| **Three evaluation points** | `THREE_POINTS_EVALUATION.md` ⭐ |
| Full methodology | `README_clinical_relevance.md` |
| Technical details | `SUMMARY.md` |
| File reference | `INDEX.md` |
| Setup check | Run `test_setup.py` |

## Key Features

- ✅ **Automated**: Evaluates 18 configurations automatically
- ✅ **Objective**: Uses BioBERT semantic similarity
- ✅ **Comprehensive**: 3 vocabularies, multiple metrics
- ✅ **Visual**: Publication-ready plots
- ✅ **Validated**: Setup test confirms prerequisites
- ✅ **Fast**: GPU acceleration, embedding caching
- ✅ **Documented**: 5 detailed documentation files

## Methodology

### BioBERT Semantic Similarity
- Model: `dmis-lab/biobert-v1.1` (pre-trained on biomedical text)
- Metric: Cosine similarity of [CLS] token embeddings
- Threshold: 0.7 for concept mapping

### Statistical Analysis
- Mean/median maximum similarity per vocabulary
- Coverage: % concepts above threshold
- Comparative analysis: Clinical vs General

### Hypothesis
Generated concepts should be more similar to clinical vocabularies (Derm7pt, Dermlike) than general medical terms (ICD-10-CM), proving clinical relevance and domain specificity.

## Requirements

```bash
pip install -r eval/requirements.txt
```

**Dependencies**: torch, transformers, numpy, pandas, matplotlib, seaborn, tqdm

**Hardware**: 
- GPU recommended (10x faster)
- 4GB RAM minimum (8GB recommended)
- ~500MB disk for BioBERT model

## Output Examples

### Summary Statistics (CSV)
| k | n | c | derm7pt_mean | dermlike_mean | icd10cm_mean |
|---|---|---|--------------|---------------|--------------|
| 3 | 2 | 50 | 0.6842 | 0.6523 | 0.4156 |
| ... | ... | ... | ... | ... | ... |

### Concept Mapping (JSON)
```json
{
  "Irregular borders": [
    {"concept": "has irregular borders", "similarity": 0.89},
    {"concept": "Ill-defined borders", "similarity": 0.82}
  ]
}
```

### Visualizations (PNG)
- Box plot: Similarity distribution by vocabulary
- Line plot: Similarity vs number of concepts
- Heatmap: Configuration performance
- Bar chart: Mapping coverage

## Usage Examples

### Default Configuration
```bash
./eval/run_clinical_relevance.sh
```

### Custom Threshold
```bash
./eval/run_clinical_relevance.sh --threshold 0.65
```

### Python Direct
```bash
python eval/concept_mapping_emb.py --dataset ham10000 --threshold 0.7
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r eval/requirements.txt` |
| Out of memory | Edit concept_mapping_emb.py, use CPU |
| Missing files | Run `test_setup.py` to diagnose |
| Slow execution | Use GPU if available |

## Next Steps

1. ✅ **Run Setup Test**: `python eval/test_setup.py`
2. 🔄 **Run Evaluation**: `./eval/run_clinical_relevance.sh`
3. 📊 **Analyze Results**: `python eval/analyze_clinical_relevance.py`
4. 📝 **Review Output**: Check `clinical_relevance/ham10000/`
5. 📈 **For Paper**: Use summary.csv and plots/

## Support

- **Setup issues**: Run `test_setup.py` for diagnosis
- **Usage questions**: See `QUICKSTART.md`
- **Technical details**: See `SUMMARY.md`
- **File reference**: See `INDEX.md`

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{clinical_relevance_eval,
  title={Clinical Relevance Evaluation Framework},
  author={Your Name},
  year={2025},
  description={BioBERT-based semantic similarity evaluation for clinical concept validation}
}
```

## Status

✅ **Implementation Complete**  
✅ **Setup Verified**  
✅ **Documentation Complete**  
✅ **Ready to Run**

---

**Get started**: `python eval/test_setup.py`

For detailed instructions, see `QUICKSTART.md`

