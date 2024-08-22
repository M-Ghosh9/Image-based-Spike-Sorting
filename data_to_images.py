# data_to_images.py

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_spike_images(signals, labels, output_dir, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = np.unique(labels)

    # Use a limited number of workers to prevent overloading memory
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for label in unique_labels:
            label_str = str(label)
            label_dir = os.path.join(output_dir, label_str)
            os.makedirs(label_dir, exist_ok=True)

            indices = np.where(labels == label)[0]
            futures = [executor.submit(save_image, signals[i], label_dir, i) for i in indices]

            for future in futures:
                try:
                    future.result()  # This will raise an exception if one occurred in save_image
                except Exception as e:
                    logging.error(f"Error processing image: {e}")

def save_image(signal, label_dir, index):
    try:
        if np.all(signal == 0) or len(signal) == 0:  # Check for empty or zero-filled signal
            print(f"Skipping empty plot: {index}")
            return

        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        ax.plot(signal)
        ax.axis('off')

        img_path = os.path.join(label_dir, f"{index}.png")
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, format='png')
        plt.close(fig)  # Close the figure after saving to free memory
        plt.close('all')  # Ensure all figures are closed

    except Exception as e:
        logging.error(f"Error saving image {index} in {label_dir}: {e}")
        plt.close('all')  # Attempt to close all figures in case of error

if __name__ == "__main__":
    try:
        signals, labels = torch.load('signals.pt')
        signals = signals.squeeze().cpu().numpy()
        logging.info("Loaded signals and labels successfully.")
    except Exception as e:
        logging.error(f"Error loading signals and labels: {e}")
        raise

    IMG_OUTPUT_DIR = 'spike_images'

    logging.info("Converting spikes to images...")
    try:
        save_spike_images(signals, labels, IMG_OUTPUT_DIR, max_workers=4)
        logging.info(f"Spike images saved to {IMG_OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"Error converting spikes to images: {e}")

