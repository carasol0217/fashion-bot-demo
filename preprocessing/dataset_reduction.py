import os
import shutil

def dataset_reduction(input_dir, output_dir, categories_to_keep):
    """
    This is for sampling the entire dataset if training on limited hardware.
    You can select which articles of clothing to train on and scope down the data.
    
    :param input_dir: Path to the full dataset.
    :param output_dir: Path where the selected categories will be saved.
    :param categories_to_keep: List of categories to copy (e.g., ['bag', 'shoes', 'dress']).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for category in categories_to_keep:
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        
        if os.path.isdir(category_path):
            shutil.copytree(category_path, output_category_path)

input_dataset_dir = '../data/'
output_dataset_dir = '../sampled_data/'
categories_to_keep = ['shoes', 'top', 'pants', 'outwear', 'necklace', 'eyewear']  # you can specify which categories if training on limited hardware
dataset_reduction(input_dataset_dir, output_dataset_dir, categories_to_keep)
