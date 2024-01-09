import numpy as np
import cv2
import random
import os

# Define image dimensions
width, height = 256, 256
num_shapes = 3
num_images = 500  # Number of images to generate

# Create a directory to save the images and masks
# output_dir = "train_images_with_masks"
output_dir = "val_images_with_masks"
output_dir_images = output_dir + "/images"
output_dir_masks = output_dir + "/masks"


os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)



# Define shapes 
shapes = ['circle', 'rectangle','triangle']
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

for image_index in range(num_images):
    # Create a blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a segmentation mask for the visible triangle
    mask = np.zeros((height, width), dtype=np.uint8)

    # Create three non-overlapping shapes for each image
    for i in range(num_shapes):
        # Select a shape (e.g., circle, rectangle, or triangle)
        shape_type = shapes[i]

        # Randomly select a color
        # color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        color = colors[i]

        # Randomly generate position and size for the shape
        x = random.randint(30, width - 30)
        y = random.randint(30, height - 30)
        size = random.randint(30, 80)

        # Draw the shape on the image
        if shape_type == 'circle':
            cv2.circle(img = image, center=(x, y), radius=size, color=color, thickness=-1)
        elif shape_type == 'rectangle':
            x2, y2 = x + size, y + size
            cv2.rectangle(image, (x, y), (x2, y2), color, -1)
        elif shape_type == 'triangle':
            points = np.array([[x, y], [x + size, y], [x + size // 2, y - size]], np.int32)
            cv2.fillPoly(image, [points], color)
            cv2.fillPoly(mask, [points], 255)  # Set the mask for the triangle to 255

    # Save the generated image and mask
    image_filename = os.path.join(output_dir_images, f'image_{image_index}.png')
    mask_filename = os.path.join(output_dir_masks, f'image_{image_index}.png')
    cv2.imwrite(image_filename, image)
    cv2.imwrite(mask_filename, mask)

print(f'{num_images} images with masks for the visible triangles generated and saved in the {output_dir} directory.')
