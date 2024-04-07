import os
from collections import defaultdict

def count_images_in_folders(root_dir):
    image_counts = defaultdict(int)

    # Traverse through all folders and subfolders
    for folder_name, _, files in os.walk(root_dir):
        image_count = sum(1 for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg')))
        image_counts[folder_name] = image_count

    return image_counts

def write_dict_to_file(dictionary, filename):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write(f'{key}: {value}\n')

# Specify the root directory containing the folders
root_directory = 'Predicted_output'

# Count the number of images in each folder
image_counts = count_images_in_folders(root_directory)

# Print the dictionary data
for folder, count in image_counts.items():
    print(f'{folder}: {count}')

# Write the dictionary data to a text file
output_file = 'image_counts.txt'
write_dict_to_file(image_counts, output_file)
print(f'Dictionary data written to {output_file}')
