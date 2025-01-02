import numpy as np
import json
import os
from PIL import Image

# Simulating an image with dimensions (3, 3072, 512)
# Assuming random values for the sake of demonstration
# Take image from folder

def image_to_numpy(folder_path):
    """
    Loads images from a folder and converts them into NumPy arrays.

    Parameters:
        folder_path (str): Path to the folder containing the images.

    Returns:
        dict: A dictionary with image filenames as keys and NumPy arrays as values.
    """
    image_arrays = {}
    image_pos = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Add supported formats
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)  # Load the image
            image_array = np.array(image)  # Convert to a NumPy array
            image_arrays[filename[:-4]] = image_array
            image_pos[filename[:-4]] = data[filename[:-4]]['eval_pos']
    return image_arrays, image_pos

# image = np.random.randint(0, 256, (3, 3072, 512), dtype=np.uint8)
folder = "./results_wide_cropped/"
os.makedirs(folder, exist_ok=True)
# Function to crop the image based on given positions
def crop_image(image_arrays, positions_arrays, crop_size):
    """
    Crops the image at specified positions.

    Parameters:
        image (numpy.ndarray): The input image with dimensions (channels, width, height).
        positions (list of int): The starting positions of the crops along the width.
        crop_size (tuple): The dimensions (width, height) of the cropped regions.

    Returns:
        list: A list of cropped images.
    """
    

    for k in image_arrays.keys():
        image = image_arrays[k]
        pos = positions_arrays[k]
        print(image.shape)
        cropped_images = []
        for start in pos:
            end = start + crop_size[0]  # Calculate the end position
            # print()
            if end <= image.shape[1]:  # Ensure the crop is within bounds
                cropped_image = image[:, start:end, :]  # Crop along the width
                print(cropped_image.shape)
                # cropped_image_transposed = cropped_image.transpose(2, 1, 0)
                # print(cropped_image_transposed.shape)
                # print(cropped_image_transposed.dtype)
                image_pil = Image.fromarray(cropped_image)
                image_path = os.path.join(folder, f"{k}_{start}.png")
                image_pil.save(image_path)

                cropped_images.append(cropped_image)
            else:
                print(f"Crop starting at {start} exceeds image width, skipping.")
    return cropped_images


wide_image_config = './data/wide_image_prompts.json'
with open(wide_image_config, 'r') as file:
    data = json.load(file)


# Define crop positions and size
# crop_positions = [0, 512, 1024, 1536, 2048, 2560]  # Example positions
crop_size = (512, 512)  # Width and height of each crop


folder_path = './results_wide'
image_arrays, image_crop_positions = image_to_numpy(folder_path)
print(image_crop_positions)



# Perform cropping
cropped_images = crop_image(image_arrays, image_crop_positions, crop_size)

# Output the number of cropped images
# len(cropped_images)
