# Unsupervised Concept Discovery for Interpretable Medical Image Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Official implementation for the thesis**: *Cluster-Based Unsupervised Concept Discovery for Interpretable Medical Image Classification*

This repository contains the complete pipeline for discovering clinically relevant visual concepts from dermatoscopic images without expert annotations, and using these concepts to build interpretable Concept Bottleneck Models (CBMs) for melanoma detection.

## Overview

Standard Concept Bottleneck Models require extensive expert annotations, creating a significant bottleneck for deploying interpretable AI in healthcare. This work addresses this limitation by developing an automated four-stage pipeline:

1. **Vision Encoder**: Fine-tuned EfficientNetV2-S extracts semantic embeddings from dermatoscopic images
2. **Clustering**: UMAP dimensionality reduction + K-means identifies natural visual groupings  
3. **Vision-Language Description**: Gemini 2.5 Flash generates rich descriptions of cluster characteristics
4. **Concept Extraction**: LLMs extract structured concept vocabularies grounded in actual visual patterns

**Key Results**: The best-performing configuration achieved **78.15% validation accuracy** on melanoma classification, outperforming the LaBo baseline by **+8.2% relative improvement** while maintaining full interpretability through 200 automatically discovered concepts.

## Repository Structure

```
github/
├── pipeline/               # Four-stage concept discovery pipeline
│   ├── dnn/               # Stage 1: Vision encoder training
│   ├── concept_extractor/ # Stages 2-4: Clustering and concept discovery
│   └── cbm/               # Concept Bottleneck Model training (LaBo)
└── eval/                  # Concept quality evaluation framework
    ├── clinical_relevance/ # BioBERT-based semantic alignment evaluation
    ├── manual_concepts/    # Reference vocabularies (Derm7pt, Dermlike)
    └── general_concepts/   # Control vocabulary (ICD-10-CM)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- Conda (recommended for environment management)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd github

# Create conda environment
conda create -n concept-cbm python=3.8
conda activate concept-cbm

# Install dependencies for each component
pip install -r pipeline/dnn/requirements.txt
pip install -r pipeline/concept_extractor/requirements.txt
pip install -r eval/requirements.txt
```

### Environment Setup

For Vision-Language Models (Gemini), set up your API key:

```bash
# Create .env file in pipeline/concept_extractor/
echo "GEMINI_API_KEY=your_api_key_here" > pipeline/concept_extractor/.env
```

## Pipeline Components

### 1. Vision Encoder (`pipeline/dnn/`)

Fine-tunes EfficientNetV2-S on HAM10000 dataset for melanoma classification and extracts semantic embeddings.

```bash
cd pipeline/dnn

# Train vision encoder
python train.py --dataset ham10000 \
                --model efficientnetv2_s \
                --epochs 50 \
                --batch_size 32

# Extract embeddings
python eval.py --dataset ham10000 \
               --checkpoint weights/best_model.pth \
               --extract_embeddings
```

**Output**: Pre-activation embeddings saved to `embeddings/ham10000/{train,val,test}.pkl`

### 2. Concept Extractor (`pipeline/concept_extractor/`)

Performs clustering, generates descriptions, and extracts concepts.

#### Stage 2: Clustering

```bash
cd pipeline/concept_extractor

# Apply UMAP + K-means clustering
python clustering.py --dataset ham10000 \
                    --k 5 \
                    --d 3 \
                    --split train

# Select representative images
python select_representative.py --dataset ham10000 \
                               --k 5 \
                               --n 4
```

#### Stage 3-4: Description & Concept Extraction

```bash
# Complete pipeline for multiple configurations
./run_pipeline_gemini.sh
```

This runs:
- Paragraph generation (Gemini 2.5 Flash)
- Concept extraction for various hyperparameters (k, n, c)

**Key hyperparameters**:
- `k`: Number of clusters (1, 3, 5, 10)
- `n`: Representatives per cluster (1, 2, 4)
- `c`: Concepts to extract (1, 3, 5, 10, 30, 50, 100, 200)

**Output**: Concept vocabularies in `concepts/gemini/ham10000/concept_all_{k}_{n}_{c}.json`

### 3. Concept Bottleneck Model (`pipeline/cbm/`)

Trains interpretable classifiers using discovered concepts via LaBo framework.

```bash
cd pipeline/cbm

# Prepare LaBo-compatible datasets
python prepare_labo_data.py --k 5 --n 4 --c 200 --vlm gemini

# Train CBM
python train_labo.py --k 5 --n 4 --c 200 --vlm gemini
```

**Output**: Trained models and results in `results/ham10000_k{k}_n{n}_c{c}_gemini/`

### 4. Evaluation Framework (`eval/`)

Evaluates concept quality through three dimensions:

#### Clinical Relevance
Measures semantic alignment with dermatological vocabularies using BioBERT embeddings:

```bash
cd eval

# Run clinical relevance evaluation
python concept_mapping_emb.py --dataset ham10000

# Generate analysis and visualizations
python analyze_clinical_relevance.py
```

**Metrics**:
- Mean similarity to Derm7pt/Dermlike (clinical vocabularies)
- Comparison with ICD-10-CM (general medical codes)
- Mapping coverage and specificity scores

#### Concept Similarity Analysis

```bash
# Compute concept-paragraph similarity
python similarity.py  # Processes all k-n configurations

# Visualize trends
python trend.py
```

**Output**: Similarity distributions and trend plots in `eval/`

## Key Results

### Concept Quality

Generated concepts demonstrated strong clinical relevance:

- **Mean similarity to clinical vocabularies**: 0.88-0.93 (Derm7pt/Dermlike)
- **Concepts exceeding threshold**: 97.5%
- **Specificity score**: 1.84-3.80 average mappings (focused, not generic)
- **Faithfulness score**: 0.88-0.89 (grounded in visual descriptions)

### Classification Performance

Best configuration (k=5, n=4, c=200):

- **Validation Accuracy**: 78.15%
- **F1-Score**: 0.756
- **Improvement over LaBo baseline**: +5.93 percentage points (+8.2% relative)
- **Gap to black-box model**: 6.25 percentage points

### Key Findings

1. **Vocabulary size matters**: 200 concepts dramatically outperform smaller vocabularies
2. **Visual grounding is critical**: Cluster-based concepts outperform linguistic priors
3. **Semantic quality metrics are necessary but insufficient**: High similarity scores don't guarantee downstream performance
4. **Annotation-free approach is viable**: Demonstrates clinically relevant concepts can be discovered without expert labels

## Reproducing Thesis Results

### Complete Pipeline Run

```bash
# 1. Train vision encoder
cd pipeline/dnn
./run_all_training.sh

# 2. Generate concepts for all configurations
cd ../concept_extractor
./run_pipeline_gemini.sh

# 3. Train all CBM configurations
cd ../cbm
python train_labo.py --all

# 4. Evaluate concept quality
cd ../../eval
python concept_mapping_emb.py --dataset ham10000
python analyze_clinical_relevance.py
```

### Expected Runtime

- Vision encoder training: ~2-3 hours (single GPU)
- Concept extraction (96 configs): ~6-8 hours (with API rate limits)
- CBM training (96 configs): ~12-16 hours (single GPU)
- Evaluation: ~30-45 minutes

## Dataset

This work uses the **HAM10000** dataset:

> Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.

**Binary Classification Task**: Melanoma (mel) vs. Melanocytic nevi (nv)
- Training: 1,113 melanoma + 6,705 nevi images
- Validation/Test: Standard split

Download: [HAM10000 on ISIC Archive](https://www.isic-archive.com/)

Place dataset in: `../datasets/ham10000/images/`


## Acknowledgments

This implementation builds upon:

- **LaBo** (Language-Aided Bottleneck): Oikarinen et al., 2023
- **EfficientNetV2**: Tan & Le, 2021  
- **UMAP**: McInnes et al., 2018
- **BioBERT**: Lee et al., 2020
- **Gemini 2.5 Flash**: Google DeepMind

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues:
- Open an issue in this repository
---

**Note**: This repository contains the complete implementation for reproducibility and transparency. The code has been cleaned and documented for public release while maintaining full functionality of all experiments described in the thesis.

