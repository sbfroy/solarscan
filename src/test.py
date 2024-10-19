from sklearn.model_selection import train_test_split
import shutil
import os

def split_data(data_dir, test_dir, val_dir, val_split=0.5):
    categories = os.listdir(data_dir)
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        images = os.listdir(category_dir)
        
        # Split the images
        test_images, val_images = train_test_split(images, test_size=val_split)
        
        # Create directories for train and validation data if they don't exist
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        # Move the train images
        for img in test_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(test_dir, category, img))
        
        # Move the validation images
        for img in val_images:
            shutil.move(os.path.join(category_dir, img), os.path.join(val_dir, category, img))

# Example usage
data_dir = 'src/../data/images/test'  # Adjusted path to the images folder
test_dir = 'src/../data/images/test2'
val_dir = 'src/../data/images/val'
split_data(data_dir, test_dir, val_dir, val_split=0.5)