import os

import torch
import torchvision
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class Animal10Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, image_size, indices, mode):
    self.compose_transforms(image_size, mode)
    self.load_label_data(dataset_path)

    image_paths = []
    image_labels = []
    for label in os.listdir(dataset_path):
      image_names = os.listdir(os.path.join(dataset_path, label))
      image_paths += [os.path.join(dataset_path, label, image_name) for image_name in image_names]
      image_labels += [self.label2index[label]] * len(image_paths)

    self.images = np.array(image_paths)[indices]
    self.labels = np.array(image_labels)[indices]

  def __getitem__(self, index):
    image_path = self.images[index]
    image = self.transform(Image.open(image_path).convert("RGB"))
    label = torch.tensor(self.labels[index], dtype=torch.long)

    return image, label

  def __len__(self):
    return len(self.images)

  def compose_transforms(self, image_size, mode):
    if mode == "train":
      self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomCrop(image_size, padding=24),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    elif mode == "validation":
      self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif mode == "test":
      self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  def load_label_data(self, dataset_path):
      self.label2index = {label: index for index, label in enumerate(os.listdir(dataset_path))}
      self.index2label = {index: label for index, label in enumerate(os.listdir(dataset_path))}

def create_train_validation_dataloaders(dataset_path, image_size, batch_size):
  dataset_size = sum([len(os.listdir(os.path.join(dataset_path, label))) for label in os.listdir(dataset_path)])
  indices = np.arange(dataset_size)
  np.random.shuffle(indices)
  train_indices, validation_indices = train_test_split(indices)
  train_dataset = Animal10Dataset(dataset_path, image_size, train_indices, "train")
  validation_dataset = Animal10Dataset(dataset_path, image_size, validation_indices, "validation")

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
  validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

  return train_dataloader, validation_dataloader

def transform_image(image):
  transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((224, 224)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  return transform(image)
