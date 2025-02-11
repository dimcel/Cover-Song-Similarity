import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from src.logger import TensorBoardLogger
from src.loss import ContrastiveLoss

def evaluate(model, data_loader, criterion, device):
    """ Evaluates the model and returns loss, accuracy, precision, recall, and F1 score. """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            pairs, labels = batch
            pairs, labels = pairs.to(device), labels.to(device)

            input1, input2 = pairs[:, 0, :, :].unsqueeze(1), pairs[:, 1, :, :].unsqueeze(1)
            output1, output2 = model(input1, input2)
            loss = criterion(output1, output2, labels)
            total_loss += loss.item()

            # Distance-based classification
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=1)
    recall = recall_score(all_labels, all_predictions, zero_division=1)
    f1 = f1_score(all_labels, all_predictions, zero_division=1)

    return loss, accuracy, precision, recall, f1

def train_siamese(device, model, train_loader, val_loader, test_loader, epochs=50, lr=0.001, patience=5):

    print(f"Training on {device}")

    model.to(device)

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger = TensorBoardLogger()

    best_val_loss = float('inf')
    patience_counter = 0

    log_dir = f"log/{datetime.now().strftime('%Y-%m-%d')}"
    os.makedirs(log_dir, exist_ok=True)

    print("Training Started")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []

        for step, batch in enumerate(train_loader):
            pairs, labels = batch
            pairs, labels = pairs.to(device), labels.to(device)

            input1, input2 = pairs[:, 0, :, :].unsqueeze(1), pairs[:, 1, :, :].unsqueeze(1)

            optimizer.zero_grad()
            output1, output2 = model(input1, input2)
            loss = criterion(output1, output2, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Distance-based classification
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance > 0.5).float()

            train_predictions.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            logger.log_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + step)

        train_acc = accuracy_score(train_labels, train_predictions)
        avg_train_loss = total_train_loss / len(train_loader)

        logger.log_scalar("Train/Average_Loss", avg_train_loss, epoch)
        logger.log_scalar("Train/Accuracy", train_acc, epoch)

        # Evaluate on Validation Set
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)

        logger.log_scalar("Validation/Loss", val_loss, epoch)
        logger.log_scalar("Validation/Accuracy", val_acc, epoch)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        os.makedirs("models", exist_ok=True)
        # Early Stopping
        if val_loss < best_val_loss:
            print("Model improved. Saving checkpoint.")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"models/best_siamese_model_{epochs}.pth")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    logger.close()
    print("Training Complete.")
    print("Starting Evaluation on Test Set.")
    
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = (val_loss, val_acc, val_prec, val_rec, val_f1)
    test_metrics = evaluate(model, test_loader, criterion, device)
    

    # metrics to log file
    with open(f"{log_dir}/metrics.txt", "w") as f:
        f.write("Evaluation Results\n\n")

        f.write("Evaluation on Train Set:\n")
        f.write(f"Loss: {train_metrics[0]}\nAccuracy: {train_metrics[1]}\nPrecision: {train_metrics[2]}\nRecall: {train_metrics[3]}\nF1 Score: {train_metrics[4]}\n\n")

        f.write("Evaluation on Validation Set:\n")
        f.write(f"Loss: {val_metrics[0]}\nAccuracy: {val_metrics[1]}\nPrecision: {val_metrics[2]}\nRecall: {val_metrics[3]}\nF1 Score: {val_metrics[4]}\n\n")

        f.write("Evaluation on Test Set:\n")
        f.write(f"Loss: {test_metrics[0]}\nAccuracy: {test_metrics[1]}\nPrecision: {test_metrics[2]}\nRecall: {test_metrics[3]}\nF1 Score: {test_metrics[4]}\n")

    print(f"Metrics saved to {log_dir}/metrics.txt")




# def train_siamese(device, model, train_loader, val_loader, test_loader, epochs=50, lr=0.001, patience=5):
#     print(f"Training on {device}")

#     model.to(device)

#     criterion = ContrastiveLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     logger = TensorBoardLogger()

#     best_val_loss = float('inf')
#     patience_counter = 0

#     log_dir = f"log/{datetime.now().strftime('%Y-%m-%d')}"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, "training_log.txt")

#     print("Training Started")

#     with open(log_file, "a") as f:  # Open file in append mode
#         f.write("## Training Log ##\n")
#         f.write(f"Training started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Device: {device}\n\n")

#     for epoch in range(epochs):
#         start_time = time.time()

#         model.train()
#         total_train_loss = 0
#         train_predictions = []
#         train_labels = []

#         for step, batch in enumerate(train_loader):
#             pairs, labels = batch
#             pairs, labels = pairs.to(device), labels.to(device)

#             input1, input2 = pairs[:, 0, :, :].unsqueeze(1), pairs[:, 1, :, :].unsqueeze(1)

#             optimizer.zero_grad()
#             output1, output2 = model(input1, input2)
#             loss = criterion(output1, output2, labels)

#             loss.backward()
#             optimizer.step()

#             total_train_loss += loss.item()

#             # Distance-based classification
#             euclidean_distance = nn.functional.pairwise_distance(output1, output2)
#             predictions = (euclidean_distance > 0.5).float()

#             train_predictions.extend(predictions.cpu().numpy())
#             train_labels.extend(labels.cpu().numpy())

#             logger.log_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + step)

#         train_acc = accuracy_score(train_labels, train_predictions)
#         avg_train_loss = total_train_loss / len(train_loader)

#         logger.log_scalar("Train/Average_Loss", avg_train_loss, epoch)
#         logger.log_scalar("Train/Accuracy", train_acc, epoch)

#         # Evaluate on Validation Set
#         val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)

#         logger.log_scalar("Validation/Loss", val_loss, epoch)
#         logger.log_scalar("Validation/Accuracy", val_acc, epoch)

#         epoch_time = time.time() - start_time
#         log_message = (
#             f"Epoch [{epoch+1}/{epochs}] | "
#             f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
#             f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
#             f"Time: {epoch_time:.2f} sec"
#         )

#         print(log_message)

#         # Save log to file
#         with open(log_file, "a") as f:
#             f.write(log_message + "\n")

#         os.makedirs("models", exist_ok=True)

#         # Early Stopping
#         if val_loss < best_val_loss:
#             print("Model improved. Saving checkpoint.")
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), f"models/best_siamese_model_{epochs}.pth")

#             with open(log_file, "a") as f:
#                 f.write("Model improved. Saving checkpoint.\n")
#         else:
#             patience_counter += 1
#             patience_message = f"Early stopping patience: {patience_counter}/{patience}"
#             print(patience_message)

#             with open(log_file, "a") as f:
#                 f.write(patience_message + "\n")

#             if patience_counter >= patience:
#                 print("Early stopping triggered!")

#                 with open(log_file, "a") as f:
#                     f.write("Early stopping triggered!\n")
#                 break

#     with open(log_file, "a") as f:
#         f.write("## Training Completed ##\n")
