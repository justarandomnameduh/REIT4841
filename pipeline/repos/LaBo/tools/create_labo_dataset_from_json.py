"""
Create LaBo dataset files (concepts npy, cls names, concept2cls, class2concepts.json, splits)
from a concept_extractor JSON like pipeline/concept_extractor/ham10000_derm7pt.json

Run this from the repo root:
  python pipeline/repos/LaBo/tools/create_labo_dataset_from_json.py

It will create: datasets/ham10000_derm7pt/concepts/* and datasets/ham10000_derm7pt/splits/*
and try to symlink images to datasets/ham10000/images if present.
"""

import json
import numpy as np
import pickle
from pathlib import Path
import shutil
import sys

ROOT = Path('/home/nqmtien/REIT4841')
SRC_JSON = ROOT / 'pipeline' / 'concept_extractor' / 'ham10000_derm7pt.json'
OUT_CONCEPT_DIR = ROOT / 'datasets' / 'ham10000_derm7pt' / 'concepts'
OUT_SPLITS_DIR = ROOT / 'datasets' / 'ham10000_derm7pt' / 'splits'
OUT_DATASET_DIR = ROOT / 'datasets' / 'ham10000_derm7pt'

def main():
    if not SRC_JSON.exists():
        print(f"Error: source JSON not found: {SRC_JSON}")
        sys.exit(1)

    data = json.loads(SRC_JSON.read_text())
    all_concepts = data.get('all', [])
    class_concepts = data.get('class_concepts', {})

    # Ensure output dirs
    OUT_CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Save class2concepts.json (use class_concepts as-is)
    with open(OUT_CONCEPT_DIR / 'class2concepts.json', 'w') as f:
        json.dump(class_concepts, f, indent=2)

    # Build concepts_raw, cls_names and concept2cls
    cls_names = sorted(list(class_concepts.keys()))
    all_list = []
    concept2cls = []

    # maintain ordering: iterate cls_names then their concepts
    for cls_idx, cls in enumerate(cls_names):
        concepts = class_concepts.get(cls, [])
        for concept in concepts:
            all_list.append(concept)
            concept2cls.append(cls_idx)

    # Fall back to global 'all' if class-specific mapping empty
    if not all_list and all_concepts:
        all_list = all_concepts
        # try assign to first class repeatedly (fallback)
        concept2cls = [0] * len(all_list)
        print("Warning: class-specific concepts empty; using global 'all' as concepts")

    # Save npy files
    np.save(OUT_CONCEPT_DIR / 'concepts_raw.npy', np.array(all_list, dtype=object))
    np.save(OUT_CONCEPT_DIR / 'cls_names.npy', np.array(cls_names, dtype=object))
    np.save(OUT_CONCEPT_DIR / 'concept2cls.npy', np.array(concept2cls, dtype=np.int32))

    print(f"Wrote concepts: {len(all_list)} concepts for {len(cls_names)} classes")

    # Create empty splits (so LaBo can start); user can replace with real splits later
    class2images = {cls: [] for cls in cls_names}
    for split in ['train', 'val', 'test']:
        p = OUT_SPLITS_DIR / f'class2images_{split}.p'
        with open(p, 'wb') as f:
            pickle.dump(class2images, f)

    print(f"Wrote empty split pickles to {OUT_SPLITS_DIR}")

    # Try create symlink to ham10000 images if available
    target_images = ROOT / 'datasets' / 'ham10000' / 'images'
    link = OUT_DATASET_DIR / 'images'
    if target_images.exists():
        try:
            if link.exists() or link.is_symlink():
                print(f"Images link already exists: {link}")
            else:
                link.symlink_to(target_images)
                print(f"Created symlink: {link} -> {target_images}")
        except Exception as e:
            print(f"Warning: could not create symlink: {e}")
    else:
        print(f"Note: ham10000 images not found at {target_images}; please add images under {OUT_DATASET_DIR}/images/")

    print("Done.")

if __name__ == '__main__':
    main()
