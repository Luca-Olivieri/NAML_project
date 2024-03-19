import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import transforms

from pathlib import Path

# Define a custom dataset class
class CustomDataset(Dataset):
   def __init__(self, images, labels, transform=None):
      self.images = images
      self.labels = labels
      self.transform = transform

   def __len__(self):
      return len(self.images)

   def __getitem__(self, idx):
      image = self.images[idx]
      imageResize = cv2.resize(image, (299, 299))

      label = self.labels[idx]
      if self.transform:
         imageResize = self.transform(imageResize)
      return imageResize, label

def load_images_with_masks(image_directory, mask_directory):
    images = []
    labels = []
    for image_filename in os.listdir(image_directory):
        if image_filename.endswith(".tif"):
            image_filepath = os.path.join(image_directory, image_filename)
            mask_filename = image_filename + '.png'
            mask_filepath = os.path.join(mask_directory, mask_filename)
            try:
                image = cv2.imread(image_filepath)
                mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(mask)
            except Exception as e:
                print(f"Error loading image {image_filename} or mask: {e}")
    return np.array(images), np.array(labels)

def extract_subimages(images, labels, subimage_size=64, step_size=8):
    subimages = []
    sublabels = []
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        height, width = image.shape[:2]
        for y in range(0, height - subimage_size + 1, step_size):
            for x in range(0, width - subimage_size + 1, step_size):
                subimage = image[y:y+subimage_size, x:x+subimage_size]
                sublabel = label[y:y+subimage_size, x:x+subimage_size]
                if np.all(sublabel == 0) or np.all(sublabel == 255):
                    subimages.append(subimage)
                    sublabels.append(sublabel[0][0])  # Assuming all values are the same in the sublabel
    return np.array(subimages), np.array(sublabels)
            
def create_dataloader(dataset_path, batch_size=100):

   # Replace 'image_directory' and 'mask_directory' with the paths to your image and mask directories
   # image_directory = 'neuroendocrine_/images'
   # mask_directory = 'neuroendocrine_/masks'

   image_directory = "neuroendocrine_/images"
   mask_directory = "neuroendocrine_/masks"

   images, labels = load_images_with_masks(image_directory, mask_directory)
   train_images, train_labels = extract_subimages(images[:-1], labels)
   val_images, val_labels = extract_subimages(images[-1:], labels)

   print("Shape of the train_images array:", train_images.shape)
   print("Shape of the train_labels array:", train_labels.shape)
   print("Shape of the val_images array:", val_images.shape)
   print("Shape of the val_labels array:", val_labels.shape)

   # labels should be 0 or 1
   train_labels[train_labels == 255] = 1
   val_labels[val_labels == 255] = 1

   del images
   del labels

   del val_images
   del val_labels

   # Assuming train_images and train_labels are your training data
   # Calculate the indices of each class
   class_0_indices = np.where(train_labels == 0)[0]
   class_1_indices = np.where(train_labels == 1)[0]

   # Determine the size of the minority class
   minority_class_size = len(class_0_indices)

   # Randomly sample the same number of samples from the majority class
   undersampled_class_1_indices = np.random.choice(class_1_indices, minority_class_size, replace=False)

   del class_1_indices
   del minority_class_size

   # Concatenate the indices of both classes
   undersampled_indices = np.concatenate([class_0_indices, undersampled_class_1_indices])

   del class_0_indices
   del undersampled_class_1_indices

   # Shuffle the indices to mix the samples from both classes
   np.random.shuffle(undersampled_indices)

   # Use the undersampled indices to create a new balanced dataset
   undersampled_images = train_images[undersampled_indices]
   undersampled_labels = train_labels[undersampled_indices]

   del train_images
   del train_labels

   del undersampled_indices

   # Define transformations for resizing and normalization
   transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

   # Create the undersampled dataset
   undersampled_dataset = CustomDataset(undersampled_images, undersampled_labels, transform=transform)

   del undersampled_images
   del undersampled_labels

   # Create the data loader for the undersampled dataset
   undersampled_loader = DataLoader(undersampled_dataset, batch_size=batch_size, shuffle=True)

   del undersampled_dataset

   return undersampled_loader

def get_images(dataset_path, test=False):

   suff = None
      
   if test == False:
      suff = "images"
   else:
      suff = "masks"

   images_dir_path = Path(f"{dataset_path}/{suff}")

   # List all files in the folder
   images_paths = sorted(list(images_dir_path.iterdir()))

   return images_paths