"""
This will be used to load and also encode the sequences
"""
import numpy as np
import random
import torch


def read_utr_mrl(filename: str) -> dict[str, float]:
    """
    This function will take the lines of sequences from the sequence csv file and return the sequences as well as the MRL values
    """
    raw_lines: list[str] = []

    with open(filename, "r", encoding="utf-8-sig") as file:

        for line in file:

            raw_lines.append(line.rstrip("\n"))

    num_lines = len(raw_lines)

    utr_mrl_dict = {}

    for r in range(1, num_lines):

        split_lines = raw_lines[r].strip().split(",")

        utr = str(split_lines[1])
        mrl = float(split_lines[32])

        utr_mrl_dict[utr] = mrl

        if r == num_lines - 1:
            print(num_lines - 1, "rows read.")

    return utr_mrl_dict



def one_hot_encode(utr: str) -> np.ndarray:
    """
    This function takes a utr string and turns it into a n-dimensional array
    """

    array = np.zeros((4, 50))

    nt_order = "ATGC"

    for i in range(50):

        nt = str(utr[i])

        index = nt_order.index(nt)

        array[index, i] = 1

    return array



def encode_sequence(utr_mrl_dict: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """
    This function will go through the dictionary, encode each sequence, then return an array for the encoded sequences, and an array for the MRL values for each
    """
    num_seq = len(utr_mrl_dict)

    # maybe add to np.zeros((num_seq, 4, 50), dtype=np.float32), same for mrl_array
    encoded_arrays = np.zeros((num_seq, 4, 50), dtype=np.float32)

    mrl_array = np.zeros((num_seq), dtype=np.float32)

    count = 0

    for seq, mrl in utr_mrl_dict.items():

        seq_array = one_hot_encode(seq)

        encoded_arrays[count] = seq_array

        mrl_array[count] = mrl

        if count == num_seq - 1:
            print(count+1, "sequences encoded.")

        count += 1

    return (encoded_arrays, mrl_array)



def assign_data(encoded_data: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    This function will take a seed and get a bunch of indexes at proportions for testing, validating, and training. It will then use the indexes to randomly pull the encoded data and assign it to each group. Then, a tuple of each group will be returned   
    """

    num_seq = len(encoded_data[0])

    seed = 42
    random_num_gen = random.Random(seed)

    training_fract = 0.8
    training_num = int(num_seq * training_fract)
    training = (np.zeros((training_num, 4, 50)), np.zeros(training_num))
    training_indexes = []


    validation_fract = 0.1
    validation_num = int(num_seq * validation_fract)
    validation = (np.zeros((validation_num, 4, 50)), np.zeros(validation_num))
    validation_indexes = []


    testing_num = num_seq - training_num - validation_num
    testing = (np.zeros((testing_num, 4, 50)), np.zeros(testing_num))
    testing_indexes = []


    index_list = list(range(num_seq))


    for i in range(training_num):

        index_list_len = len(index_list)

        random_num = random_num_gen.randint(0, (index_list_len - 1))

        random_index = index_list.pop(random_num)

        training_indexes.append(random_index)

        if i == training_num - 1:
            print(i+1, "Training indexes found")


    for i in range(validation_num):

        index_list_len = len(index_list)

        random_num = random_num_gen.randint(0, (index_list_len - 1))

        random_index = index_list.pop(random_num)

        validation_indexes.append(random_index)

        if i == validation_num - 1:
            print(i+1, "Validation indexes found")


    for i in range(testing_num):

        index_list_len = len(index_list)

        random_num = random_num_gen.randint(0, (index_list_len - 1))

        random_index = index_list.pop(random_num)

        testing_indexes.append(random_index)

        if i == testing_num - 1:
            print(i+1, "Testing indexes found")


    tr_count = 0

    for index in training_indexes:

        data = encoded_data[0][index]
        mrl = encoded_data[1][index]

        training[0][tr_count] = data
        training[1][tr_count] = mrl

        tr_count += 1


    va_count = 0

    for index in validation_indexes:

        data = encoded_data[0][index]
        mrl = encoded_data[1][index]

        validation[0][va_count] = data
        validation[1][va_count] = mrl

        va_count += 1


    te_count = 0

    for index in testing_indexes:

        data = encoded_data[0][index]
        mrl = encoded_data[1][index]

        testing[0][te_count] = data
        testing[1][te_count] = mrl

        te_count += 1

    print("Data fully split.")

    return (training, validation, testing)



def data_to_tensors(data: tuple[np.ndarray, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function will transform the numpy array data into pytorch tensors
    """

    sequences = data[0]
    mrls = data[1]

    sequence_tensor = torch.from_numpy(sequences).float()
    mrl_tensor = torch.from_numpy(mrls).float()

    print("Tensors made.")

    return sequence_tensor, mrl_tensor



def make_batches(data: tuple[torch.Tensor, torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    This takes the tensors for utrs and mrls and splits them up into batches of 128
    """

    batch_size = 128

    sequences = data[0]
    mrls = data[1]

    num_sequences = len(sequences)

    index_list = list(range(num_sequences))

    batch_list = []

    for i in range(0, num_sequences, batch_size):

        batch_indexes = index_list[i:(i + batch_size)]

        sequence_batch = sequences[batch_indexes]

        mrl_batch = mrls[batch_indexes]

        batch_list.append((sequence_batch, mrl_batch))

    print("Batches list made.")

    return batch_list



if __name__ == "__main__":

    pass
