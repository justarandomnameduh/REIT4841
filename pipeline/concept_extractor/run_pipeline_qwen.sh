#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate reit4841

DATASETS=("ham10000") #  "derm7pt" "cub"
K_VALUES=(1 3 5 10)
N_VALUES=(1 2 4)
C_VALUES=(1 3 5 10 30 50)

echo "========================================"
echo "Concept Extraction Pipeline - QWEN"
echo "========================================"
echo ""
echo "Datasets: ${DATASETS[@]}"
echo "K-means clusters (k): ${K_VALUES[@]}"
echo "Representative images (n): ${N_VALUES[@]}"
echo "Concept sizes (c): ${C_VALUES[@]}"
echo ""

# Function to check what needs to be done
check_status() {
    local dataset=$1
    echo "Checking status for dataset: $dataset"
    echo "----------------------------------------"
    
    local total_expected=$((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]}))
    local existing=0
    local missing=0
    
    for k in "${K_VALUES[@]}"; do
        for n in "${N_VALUES[@]}"; do
            for c in "${C_VALUES[@]}"; do
                if [ -f "concepts/qwen/$dataset/concept_all_${k}_${n}_${c}.json" ]; then
                    existing=$((existing + 1))
                else
                    missing=$((missing + 1))
                fi
            done
        done
    done
    
    echo "Concept files status:"
    echo "  Expected: $total_expected"
    echo "  Existing: $existing"
    echo "  Missing:  $missing"
    echo ""
}

for DATASET in "${DATASETS[@]}"; do
    check_status "$DATASET"
done

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing dataset: $DATASET"
    echo "========================================"
    
    for K in "${K_VALUES[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Configuration: k=$K"
        echo "----------------------------------------"
        
        # Check if clustering already exists
        CLUSTERING_FILE="clusters/$DATASET/${K}.json"
        if [ -f "$CLUSTERING_FILE" ]; then
            echo ""
            echo "[1/5] Clustering already exists: $CLUSTERING_FILE (skipping)"
        else
            echo ""
            echo "[1/5] Running UMAP + K-means clustering..."
            python clustering.py --dataset "$DATASET" --k "$K" --split train
            
            if [ $? -ne 0 ]; then
                echo "ERROR: Clustering failed for $DATASET with k=$K"
                continue
            fi
        fi
        
        for N in "${N_VALUES[@]}"; do
            echo ""
            echo "  Processing n=$N representative images..."
            
            # Check if representative selection already exists
            REPRESENTATIVE_FILE="clusters/$DATASET/representative_${K}_${N}.json"
            if [ -f "$REPRESENTATIVE_FILE" ]; then
                echo ""
                echo "  [2/5] Representative selection already exists: $REPRESENTATIVE_FILE (skipping)"
            else
                echo ""
                echo "  [2/5] Selecting representative images..."
                python select_representative.py --dataset "$DATASET" --k "$K" --n "$N"
                
                if [ $? -ne 0 ]; then
                    echo "  ERROR: Representative selection failed for $DATASET with k=$K, n=$N"
                    continue
                fi
            fi
            
            # Check if paragraph generation already exists
            PARAGRAPH_FILE="clusters/qwen/$DATASET/paragraph_${K}_${N}.json"
            if [ -f "$PARAGRAPH_FILE" ]; then
                echo ""
                echo "  [3/5] Paragraph generation already exists: $PARAGRAPH_FILE (skipping)"
            else
                echo ""
                echo "  [3/5] Generating cluster paragraphs with Qwen..."
                python gen_paragraph_qwen.py --dataset "$DATASET" --k "$K" --n "$N"
                
                if [ $? -ne 0 ]; then
                    echo "  ERROR: Paragraph generation failed for $DATASET with k=$K, n=$N"
                    continue
                fi
            fi
            
            echo ""
            echo "  [4/5] Extracting concepts with different sizes..."
            for C in "${C_VALUES[@]}"; do
                CONCEPT_FILE="concepts/qwen/$DATASET/concept_all_${K}_${N}_${C}.json"
                if [ -f "$CONCEPT_FILE" ]; then
                    echo "    ✓ Concept extraction already exists: concept_all_${K}_${N}_${C}.json (skipping)"
                else
                    echo "    Extracting $C concepts..."
                    python gen_concepts_qwen.py --dataset "$DATASET" --k "$K" --n "$N" --c "$C"
                    
                    if [ $? -ne 0 ]; then
                        echo "    WARNING: Concept extraction failed for c=$C"
                        continue
                    fi
                fi
            done
            
            echo ""
            echo "  ✓ Completed: $DATASET with k=$K, n=$N"
            echo "    - Paragraphs: clusters/qwen/$DATASET/paragraph_${K}_${N}.json"
            echo "    - Concepts: concepts/qwen/$DATASET/concept_all_${K}_${N}_[c].json"
        done
        
        echo ""
        echo "[5/5] Cleaning up intermediate files..."
        INTERMEDIATE_DIR="clusters/$DATASET/intermediate"
        if [ -d "$INTERMEDIATE_DIR" ]; then
            echo "  Intermediate files kept in: $INTERMEDIATE_DIR"
        fi
        
        echo ""
        echo "✓ Completed all n values for: $DATASET with k=$K"
        echo "  - Clustering: clusters/$DATASET/${K}.json"
        echo "  - Representatives: clusters/$DATASET/representative_${K}_[n].json"
    done
done

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Summary:"
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "$DATASET:"
    CONCEPT_DIR="concepts/qwen/$DATASET"
    if [ -d "$CONCEPT_DIR" ]; then
        CONCEPT_COUNT=$(ls -1 "$CONCEPT_DIR"/*.json 2>/dev/null | wc -l)
        echo "  - Concept files generated: $CONCEPT_COUNT"
        ls -1 "$CONCEPT_DIR"/*.json 2>/dev/null | sed 's/^/    /'
    fi
done

echo ""
echo "All processing complete!"
