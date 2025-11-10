# Auto-generated config for k=10, n=4, c=3, vlm=gemini
# Generated: 2025-10-22 14:35:33

# Base settings
proj_name = "ham10000"  # Just the dataset name, not the full combination
concept_root = '/home/nqmtien/REIT4841/pipeline/cbm/datasets/ham10000_k10_n4_c3_gemini/concepts/'
img_split_path = '/home/nqmtien/REIT4841/pipeline/cbm/datasets/ham10000_k10_n4_c3_gemini/splits'
img_path = '/home/nqmtien/REIT4841/datasets/ham10000/images'

# Hyperparameters for run name
k_clusters = 10
n_images = 4
c_concepts = 3
vlm_model = "gemini"

concept_type = "all"
img_ext = ''
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 2

## Data loader
bs = 64  # Increased batch size for faster training
num_workers = 8  # Parallel data loading
on_gpu = True

# Concept select
num_concept = 60  # 2 classes × 10 clusters × 3 concepts = 60
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
data_root = '/home/nqmtien/REIT4841/pipeline/cbm/results/ham10000_k10_n4_c3_gemini'
work_dir = '/home/nqmtien/REIT4841/pipeline/cbm/results/ham10000_k10_n4_c3_gemini'
n_shots = "all"
