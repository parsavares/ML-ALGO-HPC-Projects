import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the directory exists
output_dir = './charts'
os.makedirs(output_dir, exist_ok=True)

# Data for the 6 configurations
config_names = [
    '1 GPU, bs=4, Adam, Stage=3',
    '1 GPU, bs=4, AdamW, Stage=2',
    '1 GPU, bs=8, AdamW, Stage=2',
    '2 GPUs, bs=4, Adam, Stage=3',
    '2 GPUs, bs=4, AdamW, Stage=2',
    '2 GPUs, bs=8, AdamW, Stage=2'
]

# Data for each metric
train_loss = [6.17, 6.17, 6.17, 6.24, 6.24, 6.86]
train_runtime = [2.98, 1.91, 1.92, 3.27, 2.68, 2.98]
train_samples_per_sec = [4.02, 6.27, 6.24, 3.66, 4.47, 4.02]
train_steps_per_sec = [1.00, 1.56, 1.56, 0.91, 1.11, 1.00]
eval_loss = [7.31, 7.31, 7.31, 7.03, 7.03, 7.38]
eval_runtime = [0.18, 0.15, 0.16, 0.26, 0.15, 0.14]
eval_samples_per_sec = [21.51, 26.51, 24.36, 15.08, 26.22, 28.22]
eval_steps_per_sec = [5.37, 6.62, 6.09, 3.77, 6.55, 7.05]

# Set up the figure and subplots
fig, axes = plt.subplots(4, 2, figsize=(16, 20))  # 4 rows, 2 columns

# Plot train loss
axes[0, 0].barh(config_names, train_loss, color='skyblue')
axes[0, 0].set_title('Train Loss')
axes[0, 0].set_xlabel('Loss')

# Plot train runtime
axes[0, 1].barh(config_names, train_runtime, color='salmon')
axes[0, 1].set_title('Train Runtime (seconds)')
axes[0, 1].set_xlabel('Time (s)')

# Plot train samples per second
axes[1, 0].barh(config_names, train_samples_per_sec, color='lightgreen')
axes[1, 0].set_title('Train Samples Per Second')
axes[1, 0].set_xlabel('Samples/Sec')

# Plot train steps per second
axes[1, 1].barh(config_names, train_steps_per_sec, color='lightskyblue')
axes[1, 1].set_title('Train Steps Per Second')
axes[1, 1].set_xlabel('Steps/Sec')

# Plot eval loss
axes[2, 0].barh(config_names, eval_loss, color='gold')
axes[2, 0].set_title('Eval Loss')
axes[2, 0].set_xlabel('Loss')

# Plot eval runtime
axes[2, 1].barh(config_names, eval_runtime, color='coral')
axes[2, 1].set_title('Eval Runtime (seconds)')
axes[2, 1].set_xlabel('Time (s)')

# Plot eval samples per second
axes[3, 0].barh(config_names, eval_samples_per_sec, color='lightcoral')
axes[3, 0].set_title('Eval Samples Per Second')
axes[3, 0].set_xlabel('Samples/Sec')

# Plot eval steps per second
axes[3, 1].barh(config_names, eval_steps_per_sec, color='mediumpurple')
axes[3, 1].set_title('Eval Steps Per Second')
axes[3, 1].set_xlabel('Steps/Sec')

# Adjust layout for better readability
plt.tight_layout()

# Save the figure as a high-resolution image for the scientific report
output_path = os.path.join(output_dir, 'training_metrics_comparison.png')
plt.savefig(output_path, dpi=300)

# Show the plots
plt.show()
