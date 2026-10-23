"""
This is going to be used to train and save the CNN
"""
import csv
import torch
from model import CNN
from data import make_batches
from analysis import analyze
from pathlib import Path

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


loss_func = torch.nn.MSELoss()


OUTCOMES_DIR = Path(__file__).resolve().parent / "outcomes"
OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_metrics(model, tensors, batch_size=128):
    """Compute dataset-wide MSE and R² using the current model weights."""
    sequences, mrls = tensors
    model.eval()
    squared_error = 0.0
    label_sum = 0.0
    label_squared_sum = 0.0
    total_samples = 0

    with torch.inference_mode():
        for start in range(0, len(mrls), batch_size):
            batch_sequences = sequences[start:start + batch_size].to(device)
            batch_mrls = mrls[start:start + batch_size].to(device).reshape(-1)
            predictions = model(batch_sequences).reshape(-1)

            actual = batch_mrls.double()
            predicted = predictions.double()
            squared_error += ((predicted - actual) ** 2).sum().item()
            label_sum += actual.sum().item()
            label_squared_sum += (actual ** 2).sum().item()
            total_samples += actual.numel()

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    mse = squared_error / total_samples
    ss_total = label_squared_sum - (label_sum ** 2 / total_samples)
    r2 = 1 - squared_error / ss_total if ss_total > 0 else float("nan")
    return mse, r2


def run_epochs(model: CNN, optimizer: torch.optim.Adam, training_tensors: tuple[torch.Tensor, torch.Tensor], analyzing_tensors: tuple[torch.Tensor, torch.Tensor], num_epochs: int):
    
    history_path = OUTCOMES_DIR / "training_history.csv"
    with history_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_r2", "val_r2", "train_mse", "val_mse"])

    best_validation_loss = float("inf")

    for epoch in range(num_epochs):

        training_batches = make_batches(training_tensors)

        model.train()

        total_loss = 0.0
        total_samples = 0

        count = 0

        for sequences, mrls in training_batches:

            batch_loss, batch_size  = run_batches(model, sequences, mrls, optimizer)

            total_loss += (batch_loss * batch_size)
            total_samples += batch_size

            if count % 100 == 0 and count != 0:
                print(f"Epoch {epoch+1}, Batch {count}, Loss: {batch_loss:.4f}")

            count += 1

        average_loss = total_loss / total_samples

        print(f"Epoch {epoch+1} completed. Average Loss: {average_loss:.4f}")

        average_validation_loss = analyze(model, analyzing_tensors, epoch)


        training_mse, training_r2 = evaluate_metrics(model, training_tensors)
        validation_mse, validation_r2 = evaluate_metrics(model, analyzing_tensors)

        with history_path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, training_r2, validation_r2,
                             training_mse, validation_mse])

        print(f"Train R²: {training_r2:.4f}, Validation R²: {validation_r2:.4f} | "
              f"Train MSE: {training_mse:.4f}, Validation MSE: {validation_mse:.4f}")

        if average_validation_loss < best_validation_loss:

            best_validation_loss = average_validation_loss

            torch.save(model.state_dict(), OUTCOMES_DIR / "best_model.pth")

            print(f"Best model saved at epoch {epoch+1}")




def run_batches(model: CNN, sequences: torch.Tensor, mrls: torch.Tensor, optimizer: torch.optim.Adam):

    sequences = sequences.to(device)
    mrls = mrls.to(device)

    optimizer.zero_grad()

    predictions = model(sequences).squeeze(1)

    loss = loss_func(predictions, mrls)

    loss.backward()

    optimizer.step()

    batch_size = len(mrls)

    return loss.item(), batch_size