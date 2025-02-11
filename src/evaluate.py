import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from src.model import SiameseNetworkWithBatchNorm

def evaluate_metrics(device, model, test_loader):
    model.eval()
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    all_labels, all_predictions = [], []

    with torch.no_grad():
        for pairs, labels in test_loader:
            pairs = pairs.to(device)
            labels = labels.to(device)

            input1, input2 = pairs[:, 0, :, :].unsqueeze(1), pairs[:, 1, :, :].unsqueeze(1)
            output1, output2 = model(input1, input2)

            euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, pos_label=1)
    recall = recall_score(all_labels, all_predictions, pos_label=1)
    f1 = f1_score(all_labels, all_predictions, pos_label=1)
    auc = roc_auc_score(all_labels, all_predictions)

    print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
