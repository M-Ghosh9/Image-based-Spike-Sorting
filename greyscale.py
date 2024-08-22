#greyscale.py

import os
import gc
import h5py
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def delete_old_files(output_file_prefix):
    """
    Deletes the old output file if it exists.
    """
    if os.path.exists(output_file_prefix):
        try:
            os.remove(output_file_prefix)
            print(f"Deleted old file: {output_file_prefix}")
        except Exception as e:
            print(f"Error deleting file {output_file_prefix}: {e}")

def process_and_save_images(image_dir, output_file, img_size=(224, 224), batch_size=100):
    """
    Processes images in batches and saves them to an HDF5 file, ensuring efficient memory and disk usage.
    """
    delete_old_files(output_file)  # Clean up old output file
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        with h5py.File(output_file, 'w') as h5f:  # Open HDF5 file for writing
            for label_dir in sorted(os.listdir(image_dir), key=lambda x: int(x)):
                label_path = os.path.join(image_dir, label_dir)
                if os.path.isdir(label_path):
                    image_paths = [os.path.join(label_path, img_file) for img_file in os.listdir(label_path)]
                    label_images = len(image_paths)
                    print(f"Processing {label_images} images in label {label_dir}")

                    for start_idx in range(0, label_images, batch_size):
                        end_idx = min(start_idx + batch_size, label_images)
                        batch_image_paths = image_paths[start_idx:end_idx]
                        images = []
                        labels = [int(label_dir)] * len(batch_image_paths)

                        for img_path in batch_image_paths:
                            try:
                                with Image.open(img_path) as image:
                                    image = transform(image)
                                    images.append(image.numpy())
                            except Exception as e:
                                print(f"Error processing image {img_path}: {e}")

                        if images:
                            images_array = np.stack(images)
                            labels_array = np.array(labels)

                            # Save to HDF5
                            group_name = f'label_{label_dir}_batch_{start_idx // batch_size}'
                            h5f.create_dataset(group_name + '/images', data=images_array, compression='gzip')
                            h5f.create_dataset(group_name + '/labels', data=labels_array, compression='gzip')
                            print(f"Saved batch {start_idx + 1} to {end_idx} for label {label_dir}")

                            del images, images_array, labels_array
                            torch.cuda.empty_cache()
                            gc.collect()

        # Verify the HDF5 file
        try:
            with h5py.File(output_file, 'r') as h5f:
                total_images = sum(len(h5f[group_name + '/images']) for group_name in h5f.keys())
                total_labels = sum(len(h5f[group_name + '/labels']) for group_name in h5f.keys())
                print(f"File verification successful. Total images: {total_images}, Total labels: {total_labels}")
        except Exception as e:
            print(f"Error verifying file: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    IMAGE_DIRECTORY = 'spike_images'
    OUTPUT_FILENAME = 'processed_images.h5'
    process_and_save_images(IMAGE_DIRECTORY, OUTPUT_FILENAME)
