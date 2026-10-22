from data import read_utr_mrl, encode_sequence, assign_data, data_to_tensors, make_batches
from model import CNN
import time

def main():

    d1_start = time.perf_counter()
    seq_dict = read_utr_mrl("data/GSM3130435_egfp_unmod_1.csv")
    arrays = encode_sequence(seq_dict)
    training, analyzing, testing = assign_data(arrays)
    training_tensors = data_to_tensors(training)
    training_batches = make_batches(training_tensors)
    d1_end = time.perf_counter()

    print("Time taken by data.py:", (d1_end - d1_start))


    ### Conv1D layer

    model = CNN()

    sequences = training_batches[0][0]
    mrls = training_batches[0][1]

    output_vals = model(sequences)

    print("Input:", sequences.shape)
    print("Output:", output_vals.shape)





if __name__ == "__main__":
    main()