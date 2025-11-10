#!/bin/bash
#
# EfficientNetV2 Training Pipeline with ImageNet Pre-training
# Updated: October 15, 2025
#
# Key Changes:
#   - All models use ImageNet pre-trained weights by default
#   - Max learning rate reduced to 1e-4 (from 1e-3)
#   - Weight decay increased to 1e-4 (from 1e-5)
#   - Dropout increased to 0.5 (from 0.2)
#   - Label smoothing 0.1 added for classification
#
# Training Strategy:
#   1. Train on HAM10000 first (class and concept models)
#   2. Use HAM10000 pre-trained weights for Derm7pt (transfer learning)
#   3. Train CUB independently (class and concept models)
#
# Usage:
#   ./run_all_training.sh                    # Train all models
#   SKIP_EXISTING=true ./run_all_training.sh # Skip models that already exist
#

set -e

CONDA_ENV="reit4841"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PIPELINE_DIR")"

HAM10000_ROOT="$PROJECT_ROOT/datasets/ham10000"
DERM7PT_ROOT="$PROJECT_ROOT/datasets/derm7pt/release_v0"
CUB_ROOT="$PROJECT_ROOT/datasets/CUB_200_2011"
WEIGHTS_DIR="$SCRIPT_DIR/weights"
PLOTS_DIR="$SCRIPT_DIR/plots"
EMBEDDINGS_DIR="$SCRIPT_DIR/embeddings"

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p "$EMBEDDINGS_DIR/ham10000"
mkdir -p "$EMBEDDINGS_DIR/derm7pt"
mkdir -p "$EMBEDDINGS_DIR/cub"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "========================================"
echo "Training EfficientNetV2 Models"
echo "with ImageNet Pre-training"
echo "========================================"
echo ""
echo "Strategy:"
echo "  1. Train HAM10000 first (with ImageNet pre-training)"
echo "  2. Transfer learning to Derm7pt"
echo "  3. Train CUB independently (with ImageNet pre-training)"
echo ""
echo "New Settings:"
echo "  - ImageNet pre-trained weights: YES (default)"
echo "  - Max learning rate: 1e-4 (reduced from 1e-3)"
echo "  - Weight decay: 1e-4 (increased from 1e-5)"
echo "  - Dropout: 0.5 (increased from 0.2)"
echo "  - Label smoothing: 0.1 (new)"
echo ""

# Step 1: Train HAM10000 Class Model
if [[ -f "$WEIGHTS_DIR/best_class_ham10000.pth" && "${SKIP_EXISTING:-false}" == "true" ]]; then
    echo "----------------------------------------"
    echo "1. Training HAM10000 - Class Prediction"
    echo "----------------------------------------"
    echo "Skipping: Model already exists at $WEIGHTS_DIR/best_class_ham10000.pth"
else
    echo "----------------------------------------"
    echo "1. Training HAM10000 - Class Prediction"
    echo "   (With ImageNet pre-training)"
    echo "----------------------------------------"
    python "$SCRIPT_DIR/train.py" \
    --dataset ham10000 \
    --task class \
    --data_root "$HAM10000_ROOT" \
    --save_path "$WEIGHTS_DIR/best_class_ham10000.pth" \
    --plot_path "$PLOTS_DIR/dnn_training_class_result_ham10000.jpg" \
    --pretrained \
    --dropout 0.5 \
    --label_smoothing 0.1 \
    --batch_size 32 \
    --epochs 250 \
    --lr 0.0001 \
    --warmup_epochs 5 \
    --weight_decay 0.0001 \
    --patience 50 \
    --num_workers 4
fi

# Step 2: Train HAM10000 Concept Model
echo ""
if [[ -f "$WEIGHTS_DIR/best_concept_ham10000.pth" && "${SKIP_EXISTING:-false}" == "true" ]]; then
    echo "----------------------------------------"
    echo "2. Training HAM10000 - Concept Prediction"
    echo "----------------------------------------"
    echo "Skipping: Model already exists at $WEIGHTS_DIR/best_concept_ham10000.pth"
else
    echo "----------------------------------------"
    echo "2. Training HAM10000 - Concept Prediction"
    echo "   (With ImageNet pre-training)"
    echo "----------------------------------------"
    python "$SCRIPT_DIR/train.py" \
    --dataset ham10000 \
    --task concept \
    --data_root "$HAM10000_ROOT" \
    --save_path "$WEIGHTS_DIR/best_concept_ham10000.pth" \
    --plot_path "$PLOTS_DIR/dnn_training_concept_result_ham10000.jpg" \
    --pretrained \
    --dropout 0.5 \
    --batch_size 32 \
    --epochs 250 \
    --lr 0.0001 \
    --warmup_epochs 5 \
    --weight_decay 0.0001 \
    --patience 50 \
    --num_workers 4
fi

# Step 3: Train Derm7pt Class Model (Transfer Learning from HAM10000)
echo ""
if [[ -f "$WEIGHTS_DIR/best_class_derm7pt.pth" && "${SKIP_EXISTING:-false}" == "true" ]]; then
    echo "----------------------------------------"
    echo "3. Training Derm7pt - Class Prediction"
    echo "----------------------------------------"
    echo "Skipping: Model already exists at $WEIGHTS_DIR/best_class_derm7pt.pth"
else
    echo "----------------------------------------"
    echo "3. Training Derm7pt - Class Prediction"
    echo "   (Transfer learning from HAM10000 class model)"
    echo "----------------------------------------"
    if [[ ! -f "$WEIGHTS_DIR/best_class_ham10000.pth" ]]; then
        echo "ERROR: HAM10000 class model not found at $WEIGHTS_DIR/best_class_ham10000.pth"
        echo "       Please train HAM10000 models first."
        exit 1
    fi
    python "$SCRIPT_DIR/train.py" \
    --dataset derm7pt \
    --task class \
    --data_root "$DERM7PT_ROOT" \
    --save_path "$WEIGHTS_DIR/best_class_derm7pt.pth" \
    --plot_path "$PLOTS_DIR/dnn_training_class_result_derm7pt.jpg" \
    --pretrained \
    --pretrained_path "$WEIGHTS_DIR/best_class_ham10000.pth" \
    --dropout 0.5 \
    --label_smoothing 0.1 \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.00005 \
    --warmup_epochs 3 \
    --weight_decay 0.0001 \
    --patience 50 \
    --num_workers 4
fi

# Step 4: Train Derm7pt Concept Model (Transfer Learning from HAM10000)
echo ""
if [[ -f "$WEIGHTS_DIR/best_concept_derm7pt.pth" && "${SKIP_EXISTING:-false}" == "true" ]]; then
    echo "----------------------------------------"
    echo "4. Training Derm7pt - Concept Prediction"
    echo "----------------------------------------"
    echo "Skipping: Model already exists at $WEIGHTS_DIR/best_concept_derm7pt.pth"
else
    echo "----------------------------------------"
    echo "4. Training Derm7pt - Concept Prediction"
    echo "   (Transfer learning from HAM10000 concept model)"
    echo "----------------------------------------"
    if [[ ! -f "$WEIGHTS_DIR/best_concept_ham10000.pth" ]]; then
        echo "ERROR: HAM10000 concept model not found at $WEIGHTS_DIR/best_concept_ham10000.pth"
        echo "       Please train HAM10000 models first."
        exit 1
    fi
    python "$SCRIPT_DIR/train.py" \
    --dataset derm7pt \
    --task concept \
    --data_root "$DERM7PT_ROOT" \
    --save_path "$WEIGHTS_DIR/best_concept_derm7pt.pth" \
    --plot_path "$PLOTS_DIR/dnn_training_concept_result_derm7pt.jpg" \
    --pretrained \
    --pretrained_path "$WEIGHTS_DIR/best_concept_ham10000.pth" \
    --dropout 0.5 \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.00005 \
    --warmup_epochs 3 \
    --weight_decay 0.0001 \
    --patience 50 \
    --num_workers 4
fi

# Step 5: Train CUB Class Model
echo ""
if [[ -f "$WEIGHTS_DIR/best_class_cub.pth" && "${SKIP_EXISTING:-false}" == "true" ]]; then
    echo "----------------------------------------"
    echo "5. Training CUB - Class Prediction"
    echo "----------------------------------------"
    echo "Skipping: Model already exists at $WEIGHTS_DIR/best_class_cub.pth"
else
    echo "----------------------------------------"
    echo "5. Training CUB - Class Prediction"
    echo "   (With ImageNet pre-training - 200 classes)"
    echo "----------------------------------------"
    python "$SCRIPT_DIR/train.py" \
    --dataset cub \
    --task class \
    --data_root "$CUB_ROOT" \
    --save_path "$WEIGHTS_DIR/best_class_cub.pth" \
    --plot_path "$PLOTS_DIR/dnn_training_class_result_cub.jpg" \
    --pretrained \
    --dropout 0.5 \
    --label_smoothing 0.1 \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.0001 \
    --warmup_epochs 5 \
    --weight_decay 0.0001 \
    --patience 50 \
    --num_workers 4
fi

# Step 6: Train CUB Concept Model
echo ""
if [[ -f "$WEIGHTS_DIR/best_concept_cub.pth" && "${SKIP_EXISTING:-false}" == "true" ]]; then
    echo "----------------------------------------"
    echo "6. Training CUB - Concept Prediction"
    echo "----------------------------------------"
    echo "Skipping: Model already exists at $WEIGHTS_DIR/best_concept_cub.pth"
else
    echo "----------------------------------------"
    echo "6. Training CUB - Concept Prediction"
    echo "   (With ImageNet pre-training - 112 concepts)"
    echo "----------------------------------------"
    python "$SCRIPT_DIR/train.py" \
    --dataset cub \
    --task concept \
    --data_root "$CUB_ROOT" \
    --save_path "$WEIGHTS_DIR/best_concept_cub.pth" \
    --plot_path "$PLOTS_DIR/dnn_training_concept_result_cub.jpg" \
    --pretrained \
    --dropout 0.5 \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.00015 \
    --warmup_epochs 5 \
    --weight_decay 0.0001 \
    --patience 50 \
    --num_workers 8
fi

echo ""
echo "========================================"
echo "Extracting Embeddings"
echo "========================================"
echo ""

echo "----------------------------------------"
echo "7. Extracting HAM10000 Embeddings (Class Model)"
echo "----------------------------------------"
python "$SCRIPT_DIR/eval.py" \
    --dataset ham10000 \
    --data_root "$HAM10000_ROOT" \
    --model_path "$WEIGHTS_DIR/best_class_ham10000.pth" \
    --output_dir "$EMBEDDINGS_DIR/ham10000" \
    --task class \
    --batch_size 128 \
    --num_workers 4

echo ""
echo "----------------------------------------"
echo "8. Extracting Derm7pt Embeddings (Class Model)"
echo "----------------------------------------"
python "$SCRIPT_DIR/eval.py" \
    --dataset derm7pt \
    --data_root "$DERM7PT_ROOT" \
    --model_path "$WEIGHTS_DIR/best_class_derm7pt.pth" \
    --output_dir "$EMBEDDINGS_DIR/derm7pt" \
    --task class \
    --batch_size 128 \
    --num_workers 4

echo ""
echo "----------------------------------------"
echo "9. Extracting CUB Embeddings (Class Model)"
echo "----------------------------------------"
python "$SCRIPT_DIR/eval.py" \
    --dataset cub \
    --data_root "$CUB_ROOT" \
    --model_path "$WEIGHTS_DIR/best_class_cub.pth" \
    --output_dir "$EMBEDDINGS_DIR/cub" \
    --task class \
    --batch_size 128 \
    --num_workers 4

echo ""
echo "========================================"
echo "All tasks completed successfully!"
echo "========================================"
echo ""
echo "Trained models saved in: $WEIGHTS_DIR"
echo "Training plots saved in: $PLOTS_DIR"
echo "Embeddings saved in: $EMBEDDINGS_DIR"
echo ""
echo "Key improvements with ImageNet pre-training:"
echo "  - Faster convergence (fewer epochs needed)"
echo "  - Better generalization (reduced overfitting)"
echo "  - Higher accuracy on validation sets"
