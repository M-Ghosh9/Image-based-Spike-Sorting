# data_to_images.py

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_spike_images(signals, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = np.unique(labels)

    # Create a ThreadPoolExecutor for parallel image saving
    with ThreadPoolExecutor() as executor:
        for label in unique_labels:
            label_str = str(label)
            label_dir = os.path.join(output_dir, label_str)
            os.makedirs(label_dir, exist_ok=True)

            indices = np.where(labels == label)[0]
            # Use list comprehension to submit tasks to the executor
            futures = [executor.submit(save_image, signals[i], label_dir, i) for i in indices]

            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()  # This will raise an exception if one occurred in save_image
                except Exception as e:
                    logging.error(f"Error processing image: {e}")

def save_image(signal, label_dir, index):
    try:
        fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
        ax.plot(signal)
        ax.axis('off')

        img_path = os.path.join(label_dir, f"{index}.png")
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, format='png')
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error saving image {index} in {label_dir}: {e}")

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
        save_spike_images(signals, labels, IMG_OUTPUT_DIR)
        logging.info(f"Spike images saved to {IMG_OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"Error converting spikes to images: {e}")

