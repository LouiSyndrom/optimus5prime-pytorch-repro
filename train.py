"""
This is going to be used to train and save the CNN
"""
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


def run_epochs(model: CNN, optimizer: torch.optim.Adam, training_tensors: tuple[torch.Tensor, torch.Tensor], analyzing_tensors: tuple[torch.Tensor, torch.Tensor], num_epochs: int):
    
    count = 0

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
