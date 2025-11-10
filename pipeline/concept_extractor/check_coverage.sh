#!/bin/bash
# Check coverage of k-n-c combinations for HAM10000 in clusters and concepts folders

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Hyperparameter values
K_VALUES=(1 3 5 10)
N_VALUES=(1 2 4)
C_VALUES=(1 3 5 10 30)
VLMS=("gemini" "qwen")

# Base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_DIR="$SCRIPT_DIR/clusters/ham10000"
CONCEPTS_BASE="$SCRIPT_DIR/concepts"

echo "================================================================================"
echo "HAM10000 Coverage Check - Clusters and Concepts"
echo "================================================================================"
echo ""
echo "Checking for all combinations of:"
echo "  k: ${K_VALUES[@]}"
echo "  n: ${N_VALUES[@]}"
echo "  c: ${C_VALUES[@]}"
echo "  VLMs: ${VLMS[@]}"
echo ""
echo "Total combinations per VLM: $((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]}))"
echo "Total combinations (both VLMs): $((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]} * ${#VLMS[@]}))"
echo ""

# ============================================================================
# Check Clusters (k-dependent only, no n or c)
# ============================================================================
echo "================================================================================"
echo "1. CLUSTER FILES CHECK (clusters/ham10000/)"
echo "================================================================================"

cluster_total=${#K_VALUES[@]}
cluster_found=0
cluster_missing=()

echo "Expected cluster files: ${K_VALUES[@]}.json"
echo ""

for k in "${K_VALUES[@]}"; do
    cluster_file="$CLUSTER_DIR/${k}.json"
    
    if [ -f "$cluster_file" ]; then
        # Count number of clusters in the file
        num_clusters=$(jq '. | length' "$cluster_file" 2>/dev/null || echo "?")
        printf "${GREEN}✓${NC} k=${k}: ${cluster_file##*/} (${num_clusters} clusters)\n"
        cluster_found=$((cluster_found + 1))
    else
        printf "${RED}✗${NC} k=${k}: ${cluster_file##*/} - NOT FOUND\n"
        cluster_missing+=("k=${k}")
    fi
done

echo ""
echo "Cluster Summary: ${cluster_found}/${cluster_total} found"
if [ ${#cluster_missing[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing:${NC} ${cluster_missing[@]}"
fi

# ============================================================================
# Check Concepts (k-n-c combinations, per VLM)
# ============================================================================
echo ""
echo "================================================================================"
echo "2. CONCEPT FILES CHECK (concepts/{vlm}/ham10000/)"
echo "================================================================================"

for vlm in "${VLMS[@]}"; do
    echo ""
    echo "--- VLM: ${BLUE}${vlm}${NC} ---"
    
    concepts_dir="$CONCEPTS_BASE/${vlm}/ham10000"
    
    if [ ! -d "$concepts_dir" ]; then
        echo -e "${RED}✗ Directory not found: ${concepts_dir}${NC}"
        continue
    fi
    
    vlm_total=$((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]}))
    vlm_found=0
    vlm_missing=()
    
    for k in "${K_VALUES[@]}"; do
        for n in "${N_VALUES[@]}"; do
            for c in "${C_VALUES[@]}"; do
                concept_file="$concepts_dir/concept_all_${k}_${n}_${c}.json"
                
                if [ -f "$concept_file" ]; then
                    # Get number of concepts and classes
                    num_concepts=$(jq '.all | length' "$concept_file" 2>/dev/null || echo "?")
                    num_classes=$(jq '.class_concepts | length' "$concept_file" 2>/dev/null || echo "?")
                    printf "${GREEN}✓${NC} k=${k} n=${n} c=${c}: concept_all_${k}_${n}_${c}.json (${num_concepts} concepts, ${num_classes} classes)\n"
                    vlm_found=$((vlm_found + 1))
                else
                    printf "${RED}✗${NC} k=${k} n=${n} c=${c}: concept_all_${k}_${n}_${c}.json - NOT FOUND\n"
                    vlm_missing+=("k=${k}_n=${n}_c=${c}")
                fi
            done
        done
    done
    
    echo ""
    echo "${vlm} Summary: ${vlm_found}/${vlm_total} found"
    if [ ${#vlm_missing[@]} -gt 0 ]; then
        echo -e "${YELLOW}Missing (first 10):${NC}"
        for i in {0..9}; do
            if [ $i -lt ${#vlm_missing[@]} ]; then
                echo "  - ${vlm_missing[$i]}"
            fi
        done
        if [ ${#vlm_missing[@]} -gt 10 ]; then
            echo "  ... and $((${#vlm_missing[@]} - 10)) more"
        fi
    fi
done

# ============================================================================
# Overall Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "OVERALL SUMMARY"
echo "================================================================================"

total_expected=$((cluster_total + ${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]} * ${#VLMS[@]}))
total_found=$((cluster_found))

# Count total concepts found
gemini_count=$(find "$CONCEPTS_BASE/gemini/ham10000" -name "concept_all_*.json" 2>/dev/null | wc -l)
qwen_count=$(find "$CONCEPTS_BASE/qwen/ham10000" -name "concept_all_*.json" 2>/dev/null | wc -l)
total_found=$((total_found + gemini_count + qwen_count))

echo ""
echo "Clusters:"
echo "  Expected: ${cluster_total}"
echo "  Found:    ${cluster_found}"
echo "  Missing:  $((cluster_total - cluster_found))"
echo ""
echo "Concepts (Gemini):"
echo "  Expected: $((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]}))"
echo "  Found:    ${gemini_count}"
echo "  Missing:  $((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]} - gemini_count))"
echo ""
echo "Concepts (Qwen):"
echo "  Expected: $((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]}))"
echo "  Found:    ${qwen_count}"
echo "  Missing:  $((${#K_VALUES[@]} * ${#N_VALUES[@]} * ${#C_VALUES[@]} - qwen_count))"
echo ""
echo "================================================================================"
echo "Total:"
echo "  Expected: ${total_expected} files"
echo "  Found:    ${total_found} files"
echo "  Missing:  $((total_expected - total_found)) files"

if [ $((total_expected - total_found)) -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ ALL FILES PRESENT! Pipeline is complete.${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ INCOMPLETE: Some files are missing.${NC}"
    echo ""
    echo "To generate missing files:"
    echo "  1. Clusters:  python clustering.py --dataset ham10000 --d 3 --k <k>"
    echo "  2. Concepts:  Run the full pipeline scripts"
fi

echo "================================================================================"
