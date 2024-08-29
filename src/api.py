from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import uvicorn
import os

from models import FruitClassifier
from datasets import FruitDataset

model_weight = os.environ['WEIGHT_PATH']
num_fruit_types = 18
num_fruit_colors = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

app = FastAPI()

# Define mappers
fruit_type_mapper = {'Apple': 0,
  'Avocado': 1,
  'Banana': 2,
  'Cherry': 3,
  'Corn': 4,
  'Grape': 5,
  'Lemon': 6,
  'Mandarine': 7,
  'Mango': 8,
  'Onion': 9,
  'Orange': 10,
  'Peach': 11,
  'Pear': 12,
  'Pepper': 13,
  'Pineapple': 14,
  'Potato': 15,
  'Strawberry': 16,
  'Tomato': 17}  # Your fruit_type_to_idx dictionary
color_mapper = {'Black': 0,
  'Blue': 1,
  'Brown': 2,
  'Green': 3,
  'Orange': 4,
  'Red': 5,
  'White': 6,
  'Yellow': 7}  # Your color_to_idx dictionary

inverted_fruit_type_mapper = {v: k for k, v in fruit_type_mapper.items()}
inverted_color_mapper = {v: k for k, v in color_mapper.items()}

# Load your model here (outside of the endpoint function for efficiency)
model = FruitClassifier(num_fruit_types=num_fruit_types, num_colors=num_fruit_colors, resnet_version="resnet50").to(device=device)

model.load_state_dict(torch.load(model_weight, map_location=device))
model.to(device).eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    # Transform image to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).float().to(device)

    # Inference on image
    with torch.no_grad():
        outputs_tensor = model(img_tensor)

    fruit_type_idx = int(torch.argmax(outputs_tensor[0][:, :-1]))
    fruit_color_idx = int(torch.argmax(outputs_tensor[1][:, :-1]))

    fruit_type = inverted_fruit_type_mapper[fruit_type_idx]
    fruit_color = inverted_color_mapper[fruit_color_idx]

    return {"prediction": f"{fruit_type} {fruit_color}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)