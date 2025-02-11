import torch
import yaml
from torch.utils.data import DataLoader
from src.model import SiameseNetworkWithBatchNorm
from src.dataset import HpcpDataset
from src.train import train_siamese
from src.evaluate import evaluate_metrics

with open("src/config.yml", "r") as file:
    config = yaml.safe_load(file)

device = torch.device(config["device"])

train_dataset = HpcpDataset(config["data"]["train_path_features"],config["data"]["train_path_labels"])
val_dataset = HpcpDataset(config["data"]["val_path_features"], config["data"]["val_path_labels"])
test_dataset = HpcpDataset(config["data"]["test_path_features"], config["data"]["test_path_labels"]) 

train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

model = SiameseNetworkWithBatchNorm().to(device)

train_siamese(
    device=device,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=config["training"]["epochs"],
    lr=config["training"]["learning_rate"],
    patience=config["training"]["early_stopping"]
)

evaluate_metrics(device, model, test_loader)