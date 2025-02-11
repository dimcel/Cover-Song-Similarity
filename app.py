import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from src.model import SiameseNetworkWithBatchNorm
from src.preprocessing import preprocess_input
import yaml


with open("config_api.yml", "r") as file:
    config = yaml.safe_load(file)

device = torch.device(config["device"])
model_path = config["model_path"]

model = SiameseNetworkWithBatchNorm()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

app = FastAPI()

class PredictionRequest(BaseModel):
    input1: list
    input2: list

@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        input1 = preprocess_input(request.input1)
        input2 = preprocess_input(request.input2)

        # Convert to tensor & add batch dimension
        input1 = torch.tensor(input1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        input2 = torch.tensor(input2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            output1, output2 = model(input1, input2)
            distance = F.pairwise_distance(output1, output2).item()

        return {"distance": distance, "is_cover": distance < 0.5}

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-wav/")
async def predict_wav(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # Save uploaded files temporarily
        file1_path = f"data/{file1.filename}"
        file2_path = f"data/{file2.filename}"

        with open(file1_path, "wb") as f1, open(file2_path, "wb") as f2:
            f1.write(await file1.read())
            f2.write(await file2.read())

        # Extract features
        input1 = preprocess_input(file1_path)
        input2 = preprocess_input(file2_path)

        # Convert to tensor
        input1 = torch.tensor(input1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        input2 = torch.tensor(input2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            output1, output2 = model(input1, input2)
            distance = F.pairwise_distance(output1, output2).item()

        return {"distance": distance, "is_cover": distance < 0.5}

    except Exception as e:
        return {"error": str(e)}
