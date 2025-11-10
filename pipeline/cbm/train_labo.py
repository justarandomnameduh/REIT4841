"""
Train LaBo CBM models for all hyperparameter combinations.

This script trains LaBo models using the prepared datasets from prepare_labo_data.py
"""

import argparse
import json
import sys
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
import pandas as pd

# Add LaBo to path
LABO_ROOT = Path('/home/nqmtien/REIT4841/pipeline/repos/LaBo')
sys.path.insert(0, str(LABO_ROOT))


def create_config_file(k: int, n: int, c: int, vlm: str, dataset: str = 'ham10000'):
    """
    Create LaBo config file for specific hyperparameter combination.
    """
    cbm_root = Path('/home/nqmtien/REIT4841/pipeline/cbm')
    config_dir = cbm_root / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset-specific settings
    if dataset == 'ham10000':
        num_cls = 2
        num_concept = num_cls * k * c
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    dataset_name = f'{dataset}_k{k}_n{n}_c{c}_{vlm}'
    
    # Use absolute paths
    base_path = '/home/nqmtien/REIT4841'
    
    config_content = f'''# Auto-generated config for k={k}, n={n}, c={c}, vlm={vlm}
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Base settings
proj_name = "{dataset}"  # Just the dataset name, not the full combination
concept_root = '{base_path}/pipeline/cbm/datasets/{dataset_name}/concepts/'
img_split_path = '{base_path}/pipeline/cbm/datasets/{dataset_name}/splits'
img_path = '{base_path}/datasets/{dataset}/images'

# Hyperparameters for run name
k_clusters = {k}
n_images = {n}
c_concepts = {c}
vlm_model = "{vlm}"

concept_type = "all"
img_ext = ''
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = {num_cls}

## Data loader
bs = 64  # Increased batch size for faster training
num_workers = 8  # Parallel data loading
on_gpu = True

# Concept select
num_concept = {num_concept}  # {num_cls} classes × {k} clusters × {c} concepts = {num_concept}
use_mi = True
group_select = True
concept_select_fn = None
submodular_weights = 'none'

# Weight matrix fitting
lr = 1e-4
max_epochs = 10000

# Weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

# Normalization
use_img_norm = False
use_txt_norm = False

# Class name initialization
cls_name_init = 'none'
cls_sim_prior = 'none'
remove_cls_name = False

# CLIP Backbone
clip_model = 'ViT-L/14'

# Output
data_root = '{base_path}/pipeline/cbm/results/{dataset_name}'
work_dir = '{base_path}/pipeline/cbm/results/{dataset_name}'
n_shots = "all"
'''
    
    config_file = config_dir / f'{dataset_name}.py'
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return config_file


def train_single_config(k: int, n: int, c: int, vlm: str, dataset: str = 'ham10000'):
    """
    Train LaBo model for single hyperparameter configuration.
    """
    print(f"\n{'='*80}")
    print(f"Training: k={k}, n={n}, c={c}, vlm={vlm}")
    print(f"{'='*80}")
    
    # Create config file
    config_file = create_config_file(k, n, c, vlm, dataset)
    print(f"Config: {config_file}")
    
    # Check if dataset exists
    cbm_root = Path('/home/nqmtien/REIT4841/pipeline/cbm')
    dataset_name = f'{dataset}_k{k}_n{n}_c{c}_{vlm}'
    dataset_dir = cbm_root / f'datasets/{dataset_name}'
    
    if not dataset_dir.exists():
        print(f"Error: Dataset not found: {dataset_dir}")
        print(f"Please run: python prepare_labo_data.py --k {k} --n {n} --c {c} --vlm {vlm}")
        return False, None
    
    # Verify required files
    required_files = [
        dataset_dir / 'concepts/class2concepts.json',
        dataset_dir / 'concepts/concepts_raw.npy',
        dataset_dir / 'concepts/cls_names.npy',
        dataset_dir / 'concepts/concept2cls.npy',
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"Error: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False, None
    
    # Prepare output directory
    output_dir = cbm_root / f'results/{dataset_name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training command
    labo_root = Path('/home/nqmtien/REIT4841/pipeline/repos/LaBo')
    work_dir = str(output_dir.absolute())
    cmd = [
        'conda', 'run', '-n', 'labo', '--no-capture-output',
        'python',
        str(labo_root / 'main.py'),
        '--cfg', str(config_file),
        '--work-dir', work_dir,
        '--func', 'asso_opt_main'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}")
    print(f"\nStarting training...")
    
    start_time = time.time()
    
    # Run training (suppress output)
    original_dir = os.getcwd()
    try:
        # Change to LaBo directory for training
        os.chdir(labo_root)
        
        result = subprocess.run(
            cmd,
            text=True
        )
        
        # Change back
        os.chdir(original_dir)
    
    except Exception as e:
        print(f"Error during training: {e}")
        os.chdir(original_dir)
        return False, None
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    if result.returncode == 0:
        print(f"Training completed successfully in {minutes}m {seconds}s")
        
        # Extract evaluation results from wandb logs or checkpoint
        eval_results = {}
        try:
            # Look for the best checkpoint and extract metrics
            import re
            checkpoint_files = list(output_dir.glob('epoch*.ckpt'))
            if checkpoint_files:
                # Extract val_acc from filename (format: epoch-step-val_acc.ckpt)
                best_ckpt = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                match = re.search(r'val_acc-?([\d.]+)', str(best_ckpt))
                if match:
                    eval_results['val_acc'] = float(match.group(1))
        except Exception as e:
            print(f"Warning: Could not extract evaluation results: {e}")
        
        # Save result to single JSON file
        result_data = {
            'model': f'{dataset}_k{k}_n{n}_c{c}',
            'k': k,
            'n': n,
            'c': c,
            'vlm': vlm,
            'dataset': dataset,
            'duration_seconds': elapsed,
            **eval_results
        }
        
        # Append to ham10000_result.json
        results_file = cbm_root / 'results/ham10000_result.json'
        existing_results = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        
        # Update or append
        updated = False
        for i, r in enumerate(existing_results):
            if (r.get('k') == k and r.get('n') == n and 
                r.get('c') == c and r.get('vlm') == vlm):
                existing_results[i] = result_data
                updated = True
                break
        
        if not updated:
            existing_results.append(result_data)
        
        with open(results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        return True, result_data
    else:
        print(f"Training failed (return code: {result.returncode})")
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description='Train LaBo CBM models for all hyperparameter combinations'
    )
    
    parser.add_argument('--dataset', default='ham10000',
                       help='Dataset name (default: ham10000)')
    parser.add_argument('--k', type=int, nargs='+',
                       help='Cluster values (e.g., --k 1 3 5 10)')
    parser.add_argument('--n', type=int, nargs='+',
                       help='Image per cluster values (e.g., --n 1 2 4)')
    parser.add_argument('--c', type=int, nargs='+',
                       help='Concept per cluster values (e.g., --c 1 3 5 10 30)')
    parser.add_argument('--vlm', type=str, nargs='+',
                       help='VLM models (e.g., --vlm gemini qwen)')
    parser.add_argument('--all', action='store_true',
                       help='Train all hyperparameter combinations')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already trained configurations')
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    cbm_root = Path(__file__).parent
    results_dir = cbm_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Define default hyperparameter ranges
    all_k = [1, 3, 5, 10] #  
    all_n = [1, 2, 4]
    all_c = [1, 3, 5, 10, 30, 50, 100, 200] 
    all_vlm = ['gemini'] # , 'qwen'
    
    # Determine which configurations to train
    if args.all:
        k_values = all_k
        n_values = all_n
        c_values = all_c
        vlm_values = all_vlm
    else:
        k_values = args.k if args.k else all_k
        n_values = args.n if args.n else all_n
        c_values = args.c if args.c else all_c
        vlm_values = args.vlm if args.vlm else all_vlm
    
    # Training loop
    total = len(k_values) * len(n_values) * len(c_values) * len(vlm_values)
    current = 0
    success = 0
    failed = []
    skipped = []
    results = []
    
    print(f"\n{'='*80}")
    print(f"Training {total} LaBo CBM models for {args.dataset.upper()}")
    print(f"{'='*80}")
    print(f"k values: {k_values}")
    print(f"n values: {n_values}")
    print(f"c values: {c_values}")
    print(f"vlm values: {vlm_values}")
    print(f"{'='*80}\n")
    
    overall_start = time.time()
    
    for k in k_values:
        for n in n_values:
            for c in c_values:
                for vlm in vlm_values:
                    current += 1
                    print(f"\n[{current}/{total}] ", end='')
                    
                    # Check if already trained
                    if args.resume:
                        cbm_root = Path('/home/nqmtien/REIT4841/pipeline/cbm')
                        dataset_name = f'{args.dataset}_k{k}_n{n}_c{c}_{vlm}'
                        metadata_file = cbm_root / f'results/{dataset_name}/metadata.json'
                        
                        if metadata_file.exists():
                            print(f"Skipping k={k}, n={n}, c={c}, vlm={vlm} (already trained)")
                            skipped.append((k, n, c, vlm))
                            continue
                    
                    # Train
                    success_flag, metadata = train_single_config(k, n, c, vlm, args.dataset)
                    
                    if success_flag:
                        success += 1
                        results.append(metadata)
                    else:
                        failed.append((k, n, c, vlm))
    
    overall_elapsed = time.time() - overall_start
    overall_hours = int(overall_elapsed // 3600)
    overall_minutes = int((overall_elapsed % 3600) // 60)
    
    # Summary
    print(f"Total configurations: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Total time: {overall_hours}h {overall_minutes}m")
    
    if failed:
        print(f"\nFailed configurations:")
        for k, n, c, vlm in failed:
            print(f"  - k={k}, n={n}, c={c}, vlm={vlm}")
    
    # Save summary
    cbm_root = Path('/home/nqmtien/REIT4841/pipeline/cbm')
    logs_dir = cbm_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = logs_dir / f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total': total,
        'successful': success,
        'failed': len(failed),
        'skipped': len(skipped),
        'duration_seconds': overall_elapsed,
        'failed_configs': [{'k': k, 'n': n, 'c': c, 'vlm': vlm} for k, n, c, vlm in failed],
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
