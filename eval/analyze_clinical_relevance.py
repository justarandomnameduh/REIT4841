#!/usr/bin/env python3
"""
Analyze and visualize clinical relevance evaluation results.
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def load_summary(dataset='ham10000'):
    """Load the summary CSV file."""
    base_path = Path(__file__).parent.parent
    summary_file = base_path / 'eval' / 'clinical_relevance' / dataset / 'summary.csv'
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    df = pd.read_csv(summary_file)
    return df


def print_comparative_analysis(df):
    """Print comparative analysis of vocabularies."""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: Clinical Relevance by Vocabulary")
    print("="*80)
    
    vocab_names = ['derm7pt', 'dermlike', 'icd10cm']
    
    # Overall statistics
    print("\n1. Overall Mean Similarity (across all configurations)")
    print("-" * 60)
    for vocab in vocab_names:
        col = f'{vocab}_mean_sim'
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            print(f"  {vocab.upper():12s}: {mean:.4f} ± {std:.4f}")
    
    # By number of concepts
    print("\n2. Mean Similarity by Number of Concepts (c)")
    print("-" * 60)
    for c in sorted(df['c'].unique()):
        print(f"\n  c = {c}:")
        subset = df[df['c'] == c]
        for vocab in vocab_names:
            col = f'{vocab}_mean_sim'
            if col in df.columns:
                mean = subset[col].mean()
                print(f"    {vocab.upper():12s}: {mean:.4f}")
    
    # By number of clusters
    print("\n3. Mean Similarity by Number of Clusters (k)")
    print("-" * 60)
    for k in sorted(df['k'].unique()):
        print(f"\n  k = {k}:")
        subset = df[df['k'] == k]
        for vocab in vocab_names:
            col = f'{vocab}_mean_sim'
            if col in df.columns:
                mean = subset[col].mean()
                print(f"    {vocab.upper():12s}: {mean:.4f}")
    
    # By samples per cluster
    print("\n4. Mean Similarity by Samples per Cluster (n)")
    print("-" * 60)
    for n in sorted(df['n'].unique()):
        print(f"\n  n = {n}:")
        subset = df[df['n'] == n]
        for vocab in vocab_names:
            col = f'{vocab}_mean_sim'
            if col in df.columns:
                mean = subset[col].mean()
                print(f"    {vocab.upper():12s}: {mean:.4f}")
    
    # Mapping statistics
    print("\n5. Mapping Coverage (% concepts above threshold)")
    print("-" * 60)
    for vocab in vocab_names:
        above_col = f'{vocab}_above_threshold'
        if above_col in df.columns:
            coverage = (df[above_col] / df['num_concepts'] * 100).mean()
            print(f"  {vocab.upper():12s}: {coverage:.1f}%")


def create_visualizations(df, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    vocab_names = ['derm7pt', 'dermlike', 'icd10cm']
    vocab_colors = {'derm7pt': '#2ecc71', 'dermlike': '#3498db', 'icd10cm': '#e74c3c'}
    
    # 1. Box plot: Similarity distribution by vocabulary
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_plot = []
    for vocab in vocab_names:
        col = f'{vocab}_mean_sim'
        if col in df.columns:
            for val in df[col]:
                data_for_plot.append({'Vocabulary': vocab.upper(), 'Mean Similarity': val})
    
    plot_df = pd.DataFrame(data_for_plot)
    sns.boxplot(data=plot_df, x='Vocabulary', y='Mean Similarity', ax=ax, palette=vocab_colors)
    ax.set_title('Similarity Distribution by Vocabulary', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Maximum Similarity', fontsize=12)
    ax.set_xlabel('Target Vocabulary', fontsize=12)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_by_vocabulary.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'similarity_by_vocabulary.png'}")
    plt.close()
    
    # 2. Line plot: Similarity vs number of concepts
    fig, ax = plt.subplots(figsize=(10, 6))
    for vocab in vocab_names:
        col = f'{vocab}_mean_sim'
        if col in df.columns:
            grouped = df.groupby('c')[col].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=vocab.upper(), 
                   color=vocab_colors[vocab], linewidth=2, markersize=8)
    
    ax.set_title('Mean Similarity vs Number of Concepts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Concepts (c)', fontsize=12)
    ax.set_ylabel('Mean Maximum Similarity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_vs_concepts.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'similarity_vs_concepts.png'}")
    plt.close()
    
    # 3. Heatmap: Configuration performance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, vocab in enumerate(vocab_names):
        col = f'{vocab}_mean_sim'
        if col in df.columns:
            pivot = df.pivot_table(values=col, index='k', columns='c', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                       vmin=0, vmax=1, ax=axes[idx], cbar_kws={'label': 'Similarity'})
            axes[idx].set_title(f'{vocab.upper()}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Number of Concepts (c)', fontsize=10)
            axes[idx].set_ylabel('Number of Clusters (k)', fontsize=10)
    
    plt.suptitle('Mean Similarity Heatmap by Configuration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'similarity_heatmap.png'}")
    plt.close()
    
    # 4. Coverage analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(vocab_names))
    width = 0.25
    
    metrics = ['above_threshold', 'mapped', 'total_mappings']
    metric_labels = ['Above Threshold', 'Mapped Concepts', 'Total Mappings (÷10)']
    
    for i, metric in enumerate(metrics):
        values = []
        for vocab in vocab_names:
            col = f'{vocab}_{metric}'
            if col in df.columns:
                if metric == 'total_mappings':
                    values.append(df[col].mean() / 10)  # Scale down for visibility
                else:
                    values.append(df[col].mean())
            else:
                values.append(0)
        
        ax.bar(x + i * width, values, width, label=metric_labels[i])
    
    ax.set_title('Mapping Coverage by Vocabulary', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Vocabulary', fontsize=12)
    ax.set_ylabel('Average Count', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([v.upper() for v in vocab_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'coverage_analysis.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze clinical relevance evaluation results'
    )
    parser.add_argument('--dataset', type=str, default='ham10000',
                       help='Dataset name (default: ham10000)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("Loading evaluation results...")
        df = load_summary(args.dataset)
        print(f"Loaded {len(df)} configurations")
        
        # Print analysis
        print_comparative_analysis(df)
        
        # Create visualizations
        if not args.no_plots:
            print("\n" + "="*80)
            print("GENERATING VISUALIZATIONS")
            print("="*80)
            
            base_path = Path(__file__).parent.parent
            output_dir = base_path / 'eval' / 'clinical_relevance' / args.dataset / 'plots'
            
            create_visualizations(df, output_dir)
            
            print("\n" + "="*80)
            print("Visualization plots saved successfully!")
            print("="*80)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the evaluation first:")
        print("  python eval/concept_mapping_emb.py --dataset ham10000")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

