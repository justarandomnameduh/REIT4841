_base_ = '../base.py'
# dataset
proj_name = "ham10000_derm7pt"
concept_root = 'datasets/ham10000_derm7pt/concepts/'
img_split_path = 'datasets/ham10000_derm7pt/splits'
# reuse HAM10000 images (adjust if you have a different image root)
img_path = 'datasets/ham10000/images'

concept_type = "all"
img_ext = ''
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 2  # Auto-updated from cls_names.npy

## data loader
bs = 32
on_gpu = True

# concept select
num_concept = 0  # Auto-updated from concepts_raw.npy

# weight matrix fitting
lr = 1e-4
max_epochs = 10000

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

# CLIP Backbone
clip_model = 'ViT-L/14'
