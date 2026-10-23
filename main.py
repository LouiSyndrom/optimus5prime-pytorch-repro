from data import read_utr_mrl, encode_sequence, assign_data, data_to_tensors, make_batches
from train import run_epochs
from analysis import analyze
from model import CNN
import time
import torch
import numpy as np

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ### Data

    seq_dict = read_utr_mrl("data/GSM3130435_egfp_unmod_1.csv")
    arrays = encode_sequence(seq_dict)
    training, analyzing, testing = assign_data(arrays)


    ### Training and Analyzing

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_tensors = data_to_tensors(training)
    analyzing_tensors = data_to_tensors(analyzing)

    run_epochs(model, optimizer, training_tensors, analyzing_tensors, num_epochs=10)

    ### Testing


if __name__ == "__main__":
    main()