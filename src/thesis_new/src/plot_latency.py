#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
csv_path = os.path.expanduser('~/Downloads/latency_results.csv')
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Run the bag processor first!")
    exit()

df = pd.read_csv(csv_path)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['frame_index'], df['latency_ms'], label='Gaussian Noise Latency', color='blue')

# Add the PerturbationDrive Real-Time Threshold (33.3ms for 30 FPS)
plt.axhline(y=33.3, color='red', linestyle='--', label='30 FPS Real-Time Limit (33.3ms)')

# Formatting
plt.title('Perturbation Processing Latency per Frame')
plt.xlabel('Frame Index')
plt.ylabel('Latency (ms)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Save the chart
output_chart = os.path.expanduser('~/Downloads/latency_chart.png')
plt.savefig(output_chart)
print(f"Chart saved to: {output_chart}")
plt.show()
