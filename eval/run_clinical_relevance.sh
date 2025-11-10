#!/bin/bash

# Clinical Relevance Evaluation Script
# Evaluates generated concepts against Derm7pt, Dermlike, and ICD-10-CM vocabularies

echo "=========================================="
echo "Clinical Relevance Evaluation"
echo "=========================================="
echo ""
echo "This script evaluates generated concepts using BioBERT embeddings"
echo "against three target vocabularies:"
echo "  1. Derm7pt (human-annotated dermatology concepts)"
echo "  2. Dermlike (literature-based dermatology features)"
echo "  3. ICD-10-CM (standardized medical codes)"
echo ""
echo "Scope: k=[3,5,10], n=[2,4], c=[50,100,200] (18 configurations)"
echo ""

# Default parameters
THRESHOLD=0.7
DATASET="ham10000"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --threshold FLOAT    Similarity threshold for mapping (default: 0.7)"
            echo "  --dataset STRING     Dataset name (default: ham10000)"
            echo "  --help              Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Threshold: $THRESHOLD"
echo "  Dataset: $DATASET"
echo ""

# Check if required files exist
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_SCRIPT="$SCRIPT_DIR/concept_mapping_emb.py"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Check vocabularies
MANUAL_DIR="$SCRIPT_DIR/manual_concepts"
GENERAL_DIR="$SCRIPT_DIR/general_concepts"

echo "Checking vocabularies..."
if [ -f "$MANUAL_DIR/${DATASET}_derm7pt.json" ]; then
    echo "  ✓ Derm7pt vocabulary found"
else
    echo "  ✗ Warning: Derm7pt vocabulary not found"
fi

if [ -f "$MANUAL_DIR/${DATASET}_dermlike.json" ]; then
    echo "  ✓ Dermlike vocabulary found"
else
    echo "  ✗ Warning: Dermlike vocabulary not found"
fi

if [ -f "$GENERAL_DIR/icd10cm_descriptions.txt" ]; then
    ICD_COUNT=$(wc -l < "$GENERAL_DIR/icd10cm_descriptions.txt")
    echo "  ✓ ICD-10-CM vocabulary found ($ICD_COUNT concepts)"
else
    echo "  ✗ Warning: ICD-10-CM vocabulary not found"
fi

echo ""
echo "=========================================="
echo "Starting Evaluation..."
echo "=========================================="
echo ""

# Run evaluation
python "$EVAL_SCRIPT" --threshold "$THRESHOLD" --dataset "$DATASET"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  $SCRIPT_DIR/clinical_relevance/$DATASET/"
    echo ""
    echo "Check summary.csv for aggregate statistics"
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

