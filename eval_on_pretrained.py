import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from src.model import SiameseNetworkWithBatchNorm
from src.dataset import HpcpDataset
from src.evaluate import evaluate_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Siamese Network model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the saved model file."
    )
    return parser.parse_args()

def main():

    args = parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device(config["device"])
    print(f"Using device: {device}")
    test_dataset = HpcpDataset(config["data"]["test_path_features"], config["data"]["test_path_labels"])

    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    model = SiameseNetworkWithBatchNorm().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # Evaluate the model
    evaluate_metrics(device, model, test_loader)

if __name__ == "__main__":
    main()
# python eval_on_pretrained.py --config "src/config.yml" --model "models_2/best_siamese_model_09_2.pth"