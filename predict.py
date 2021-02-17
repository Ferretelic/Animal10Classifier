import torch
import os

from animal10 import create_train_validation_dataloaders

def load_label_data(dataset_path):
    return {index: label for index, label in enumerate(os.listdir(dataset_path))}

def predict_label(images):
  translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}

  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/Animal10Dataset"
  index2label = load_label_data(dataset_path)
  model_path = "./model.pth"

  device = torch.device("cpu")
  model = torch.load(model_path).to(device)
  label = torch.argmax(torch.nn.functional.softmax(model(images)), 1)
  print(label)
  return translate[index2label[label[0].item()]]
