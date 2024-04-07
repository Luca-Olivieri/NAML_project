import os

import numpy as np
import cv2

import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision import transforms, models

import matplotlib.pyplot as plt

def import_model(model_name):
   if model_name == "inceptionv3":
      # Load InceptionV3 model pretrained on ImageNet
      model = models.inception_v3(weights='DEFAULT')
   elif model_name == "AlexNet":
      # Load AlexNet model pretrained on ImageNet
      model = models.alexnet(weights='DEFAULT')

   return model

def setup_model(model, device):
   # Set the model to evaluation mode
   model.eval()

   num_classes = 2  # Assuming CustomDataset has a 'classes' attribute

   if isinstance(model, models.inception.Inception3):
      # Freeze all the parameters   
      for param in model.parameters():
         param.requires_grad = False

      # Modify the last layer to fit your number of classes
      model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

   elif isinstance(model, models.AlexNet):
      model.classifier.add_module('7', nn.ReLU())
      model.classifier.add_module('8', nn.Linear(1000, 512))
      model.classifier.add_module('9', nn.ReLU())
      model.classifier.add_module('10', nn.Linear(512, 128))
      model.classifier.add_module('11', nn.ReLU())
      model.classifier.add_module('12', nn.Linear(128, 32))
      model.classifier.add_module('13', nn.ReLU())
      model.classifier.add_module('14', nn.Linear(32, 2))
   # Move the model to the GPU
   model = model.to(device)

def validate(model, criterion, dataloader, device):
   # model.eval()
   val_loss = 0.0
   correct_predictions = 0
   total_samples = 0
    
   for inputs, labels in dataloader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      val_loss += loss.item() * inputs.size(0)
      _, predicted = torch.max(outputs, 1)
      correct_predictions += (predicted == labels).sum().item()
      total_samples += labels.size(0)
    
   val_loss /= total_samples
   accuracy = correct_predictions / total_samples
    
   print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
   # model.train()

def train(model, optimizer, dataloader, val_dataloader, class_weights, device, 
          num_epochs=1, max_train=200, print_every=10):

   def setup_training(model):

      if isinstance(model, models.inception.Inception3):
         model.fc.train()

      elif isinstance(model, models.AlexNet):
         model.train()

   # Training loop
   def training_loop(model, optimizer, criterion, train_dataloader, val_dataloader, device, 
                  num_epochs=num_epochs, max_train=max_train, print_every=print_every):
       
      for epoch in range(num_epochs):
        
         running_loss = 0.0
        
         for i, (inputs, labels) in enumerate(train_dataloader, 1):

            # Fetch inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)

            # Initialise gradients
            optimizer.zero_grad()

            # Feedforward
            outputs = model(inputs)

            # Backpropagate
            loss = criterion(outputs, labels)
            loss.backward()

            # Update parameters
            optimizer.step()

            # Append loss
            running_loss += loss.item() * inputs.size(0)

            # Print average loss every print_every iterations
            if i % print_every == 0:
                epoch_loss = running_loss / (print_every * len(inputs))
                print(f"Iteration [{i}/{len(train_dataloader)}], Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
                running_loss = 0.0
                validate(model, criterion, val_dataloader, device)
                print("------------------")
      
            # Iteration limit
            if i >= max_train:
                break
         
         print("==========================")

   setup_training(model)
   
   # criterion = nn.CrossEntropyLoss()

   class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

   # Define your loss function with custom class weights
   criterion = nn.CrossEntropyLoss(weight=class_weights)

   training_loop(model, optimizer, criterion, dataloader, val_dataloader, device, num_epochs, max_train, print_every)

def load_params(model, local_path, project_path, version, device):
   
   model_path = None
   params_path = None
   
   if isinstance(model, models.inception.Inception3):

      model_path = "inception_v3"
      params_path = "incv3_params.pth"

   elif isinstance(model, models.AlexNet):

      model_path = "AlexNet"
      params_path = "AlexNet_params.pth"

   full_path = f"{local_path}/{params_path}"
      
   if not os.path.exists(params_path):
      # if there are no local parameters, load the selected ones
      full_path = f"{project_path}/{model_path}/{version}/{params_path}"

   # Load the saved dictionary into your model
   state_dict = torch.load(full_path, map_location=device)
   model.load_state_dict(state_dict)

def save_params(model, local_path):

   params_path = None
   
   if isinstance(model, models.inception.Inception3):
      params_path = "incv3_params.pth"

   elif isinstance(model, models.AlexNet):
      params_path = "AlexNet_params.pth"

   # Save the model state
   torch.save(model.state_dict(), f"{local_path}/{params_path}")

def predict(model, device, image):

  testImage = image

  subimage_size = 64
  step_size = 8

  height, width = testImage.shape[:2]
  outputHeight = height - height % step_size
  outputWidth = width - width % step_size
  tumorCount = np.zeros((outputHeight, outputWidth))
  count = np.zeros((outputHeight, outputWidth))

  # Move model to GPU
  model.eval()

  for row in range(0, height - subimage_size + 1, step_size):
      if row % 50 == 0:
          print('Row ' + str(row) + '/ ' + str(height))
      for col in range(0, width - subimage_size + 1, step_size):
          subimage = testImage[row:row+subimage_size, col:col+subimage_size]

          # Prepare subimage for InceptionV3
          # Resize the subimage using OpenCV
          resized_subimage = cv2.resize(subimage, (299, 299))

          # Define the transformations
          transform = transforms.Compose([
              transforms.ToTensor(),  # Convert image to tensor
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
          ])

          # Apply the transformations and move to GPU
          transformed_subimage = transform(resized_subimage).to(device)

          # Compute output on GPU
          with torch.no_grad():
              output = model(transformed_subimage.unsqueeze(0))
              label = torch.argmax(output)

          # Write to matrix
          if label == 1:
              tumorCount[row:row+subimage_size, col:col+subimage_size] += 1
          count[row:row+subimage_size, col:col+subimage_size] += 1

  # Calculate average tumor occurrence per submatrix
  avg = np.divide(tumorCount, count)

  return avg

def get_prediction(model, local_path, project_path, version):
    
   if isinstance(model, models.inception.Inception3):
      model_name = "inception_v3"

   elif isinstance(model, models.AlexNet):
      model_name = "AlexNet"
   
   if os.path.exists("avg.png"):
      avg_path = f"{local_path}/avg.png"
   else:
      avg_path = f"{project_path}/{model_name}/{version}/avg.png"

   avg = cv2.imread(avg_path)

   # Convert the image to grayscale
   avg = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
   avg = avg/255

   return avg

def beautify_output(avg):
   printAvg = avg*255
   return printAvg

def threshold_output(avg, thr=0.5):
   modelGuess = np.where(avg >= thr, 255, 0).astype(np.uint8)[:1283, :2040]
   return modelGuess

def visualise_output(printable):
   # Displaying the numpy array as grayscale
   plt.imshow(printable, cmap='gray', vmin=0, vmax=255)  # Specify vmin and vmax
   plt.axis('off')  # Turn off axis
   plt.show()

def save_prediction(pred):
      cv2.imwrite('pred_image.png', pred)