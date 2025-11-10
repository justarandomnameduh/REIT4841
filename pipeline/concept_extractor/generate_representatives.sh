#!/bin/bash

# Script to generate representative images for different k and n combinations
# k: number of clusters [1, 3, 5, 10]
# n: number of representatives per cluster [1, 2, 4]

DATASET="ham10000"
K_VALUES=(1 3 5 10)
N_VALUES=(1 2 4)

echo "Generating representative images for dataset: $DATASET"
echo "=============================================="

for k in "${K_VALUES[@]}"; do
    for n in "${N_VALUES[@]}"; do
        echo ""
        echo "Processing k=$k, n=$n..."
        python select_representative.py --dataset $DATASET --k $k --n $n
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully generated representative_${k}_${n}.json"
        else
            echo "✗ Failed to generate representative_${k}_${n}.json"
        fi
    done
done

echo ""
echo "=============================================="
echo "All representative generation tasks completed!"
echo ""
echo "Generated files:"
ls -lh clusters/$DATASET/representative_*.json 2>/dev/null || echo "No files found"
