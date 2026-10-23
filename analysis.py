from model import CNN
from data import make_batches
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


loss_func = torch.nn.MSELoss()


def analyze(model: CNN, analyzing_tensors: tuple[torch.Tensor, torch.Tensor], epoch: int) -> float:

    analyzing_batches = make_batches(analyzing_tensors)

    model.eval()

    validation_loss = 0.0
    validation_samples = 0

    with torch.no_grad():

        for sequences, mrls in analyzing_batches:

            sequences = sequences.to(device)
            mrls = mrls.to(device)

            predictions = model(sequences).squeeze(1)

            loss = loss_func(predictions, mrls)

            batch_size = len(mrls)

            validation_loss += loss.item() * batch_size
            validation_samples += batch_size

    average_validation_loss = validation_loss / validation_samples

    print(f"Epoch {epoch+1}: Validation Loss = {average_validation_loss:.4f}")

    return average_validation_loss