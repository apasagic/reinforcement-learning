import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS ===
plot_top = True  # Set to True for top N, False for bottom N
N = 5              # Number of frames to plot

# === PROCESS Q-VALUES ===
frame_q_values = np.max(q_values, axis=1)
preferred_actions = np.argmax(q_values, axis=1)

sorted_indices = np.argsort(frame_q_values)[::-1] if plot_top else np.argsort(frame_q_values)
selected_indices = sorted_indices[:N]

# === PLOT ===
plt.figure(figsize=(15, 4))
for i, idx in enumerate(selected_indices):
    plt.subplot(1, N, i + 1)

    # Get the most recent frame (grayscale)
    base = states[idx,:,:,3]  # shape: (84, 84)

    # Compute motion (difference between last and second-last frames)
    motion = states[idx,:,:,3] - states[idx,:,:,0]

    # Normalize motion to [0, 1]
    motion_norm = (motion - motion.min())

    # Show base frame in grayscale
    plt.imshow(base, cmap='gray', interpolation='none')

    # Overlay the motion heatmap
    plt.imshow(motion_norm, cmap='bwr', alpha=0.5, interpolation='none')

    q_val = frame_q_values[idx]
    action = preferred_actions[idx]
    plt.title(f"Frame {idx}\nQ={q_val:.3f}, A={action}")
    plt.axis('off')

plt.suptitle("Top Q-Value Frames" if plot_top else "Bottom Q-Value Frames", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()
