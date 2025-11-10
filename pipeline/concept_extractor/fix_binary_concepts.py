#!/usr/bin/env python3
"""
Fix existing concept JSON files to use binary classification (mel vs nv only).

This script removes the extra 5 classes (akiec, bcc, bkl, df, vasc) from all
concept files, keeping only mel and nv.
"""

import json
import argparse
from pathlib import Path


def fix_concept_file(concept_file: Path, dry_run: bool = False) -> dict:
    """
    Remove extra classes from a concept file, keeping only mel and nv.
    
    Args:
        concept_file: Path to concept JSON file
        dry_run: If True, don't write changes, just report
    
    Returns:
        Dictionary with statistics about changes
    """
    # Load the file
    with open(concept_file, 'r') as f:
        data = json.load(f)
    
    stats = {
        'file': str(concept_file),
        'classes_before': list(data.get('class_concepts', {}).keys()),
        'classes_after': [],
        'modified': False
    }
    
    # Keep only mel and nv in class_concepts
    if 'class_concepts' in data:
        original_classes = set(data['class_concepts'].keys())
        binary_classes = {'mel', 'nv'}
        
        # Check if we need to modify
        if original_classes != binary_classes:
            stats['modified'] = True
            
            # Create new class_concepts with only mel and nv
            new_class_concepts = {
                k: v for k, v in data['class_concepts'].items() 
                if k in binary_classes
            }
            
            # Ensure both classes exist
            if 'mel' not in new_class_concepts:
                new_class_concepts['mel'] = []
            if 'nv' not in new_class_concepts:
                new_class_concepts['nv'] = []
            
            data['class_concepts'] = new_class_concepts
            stats['classes_after'] = list(new_class_concepts.keys())
            
            # Write back if not dry run
            if not dry_run:
                with open(concept_file, 'w') as f:
                    json.dump(data, f, indent=2)
        else:
            stats['classes_after'] = list(original_classes)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Fix concept JSON files for binary classification'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--vlm',
        type=str,
        choices=['gemini', 'qwen', 'all'],
        default='all',
        help='Which VLM concept files to fix'
    )
    
    args = parser.parse_args()
    
    # Find all concept files
    concept_root = Path(__file__).parent / 'concepts'
    
    vlms = ['gemini', 'qwen'] if args.vlm == 'all' else [args.vlm]
    
    print("="*80)
    print("Fixing Concept Files for Binary Classification (mel vs nv)")
    print("="*80)
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()
    
    total_files = 0
    modified_files = 0
    
    for vlm in vlms:
        vlm_path = concept_root / vlm / 'ham10000'
        
        if not vlm_path.exists():
            print(f"Warning: {vlm_path} does not exist, skipping...")
            continue
        
        # Find all concept_all_*.json files
        concept_files = sorted(vlm_path.glob('concept_all_*.json'))
        
        print(f"\n{vlm.upper()} Concept Files: {len(concept_files)} found")
        print("-"*80)
        
        for concept_file in concept_files:
            total_files += 1
            stats = fix_concept_file(concept_file, dry_run=args.dry_run)
            
            if stats['modified']:
                modified_files += 1
                status = "WOULD MODIFY" if args.dry_run else "MODIFIED"
                print(f"[{status}] {concept_file.name}")
                print(f"  Before: {len(stats['classes_before'])} classes - {stats['classes_before']}")
                print(f"  After:  {len(stats['classes_after'])} classes - {stats['classes_after']}")
            else:
                print(f"[SKIP] {concept_file.name} - already binary")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files processed: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Files unchanged: {total_files - modified_files}")
    
    if args.dry_run and modified_files > 0:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
