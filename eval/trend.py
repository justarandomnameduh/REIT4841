import matplotlib.pyplot as plt

# Data for c=200: min, q1, med, q3, max
data_stats = [
    # k=1
    {'whislo': 0.27, 'q1': 0.4, 'med': 0.56, 'q3': 0.725, 'whishi': 1, 'label': 'k=1,n=1'},
    {'whislo': 0.2, 'q1': 0.415, 'med': 0.62, 'q3': 0.78, 'whishi': 1, 'label': 'k=1,n=2'},
    {'whislo': 0.42, 'q1': 0.62, 'med': 0.7, 'q3': 0.785, 'whishi': 0.98, 'label': 'k=1,n=4'},
    # k=3
    {'whislo': 0.21, 'q1': 0.46, 'med': 0.605, 'q3': 0.76, 'whishi': 1, 'label': 'k=3,n=1'},
    {'whislo': 0.32, 'q1': 0.55, 'med': 0.695, 'q3': 0.84, 'whishi': 1, 'label': 'k=3,n=2'},
    {'whislo': 0.41, 'q1': 0.68, 'med': 0.76, 'q3': 0.88, 'whishi': 1, 'label': 'k=3,n=4'},
    # k=5
    {'whislo': 0.46, 'q1': 0.68, 'med': 0.765, 'q3': 0.845, 'whishi': 1, 'label': 'k=5,n=1'},
    {'whislo': 0.4, 'q1': 0.64, 'med': 0.75, 'q3': 0.88, 'whishi': 1, 'label': 'k=5,n=2'},
    {'whislo': 0.49, 'q1': 0.69, 'med': 0.787, 'q3': 0.86, 'whishi': 1, 'label': 'k=5,n=4'},
    # k=10
    {'whislo': 0.594, 'q1': 0.77, 'med': 0.85, 'q3': 0.91, 'whishi': 1, 'label': 'k=10,n=1'},
    {'whislo': 0.6, 'q1': 0.755, 'med': 0.847, 'q3': 0.91, 'whishi': 1, 'label': 'k=10,n=2'},
    {'whislo': 0.63, 'q1': 0.79, 'med': 0.865, 'q3': 0.92, 'whishi': 1, 'label': 'k=10,n=4'},
]

fig, ax = plt.subplots(figsize=(12, 8))
ax.bxp(data_stats, showfliers=False, patch_artist=True)
ax.set_xticklabels([stat['label'] for stat in data_stats], rotation=45, ha='right')
ax.set_xlabel('k and n combinations')
ax.set_ylabel('Similarity Score')
ax.set_title('Similarity Trends for c=200')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trend.png', dpi=300, bbox_inches='tight')
plt.show()