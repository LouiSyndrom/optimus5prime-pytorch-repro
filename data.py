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

    index_list = list(range(num_seq))


    training, index_list = assign_training(encoded_data, num_seq)

    validating, index_list = assign_validating(encoded_data, index_list, num_seq)

    testing = assign_testing(encoded_data, index_list)


    print("Data fully split.")


    return (training, validating, testing)

def assign_training(encoded_data: tuple[np.ndarray, np.ndarray], num_seq: int) -> tuple[tuple[np.ndarray, np.ndarray], list[int]]:

    training_fract = 0.8
    training_num = int(num_seq * training_fract)

    seed = 42
    random_num_gen = random.Random(seed)

    training_indexes = []
    index_list = list(range(num_seq))

    training = (np.zeros((training_num, 4, 50)), np.zeros(training_num))

    for i in range(training_num):

        index_list_len = len(index_list)

        random_num = random_num_gen.randint(0, (index_list_len - 1))

        random_index = index_list.pop(random_num)

        training_indexes.append(random_index)

        if i == training_num - 1:
            print(i+1, "Training indexes found.")

        tr_count = 0

    for index in training_indexes:

        data = encoded_data[0][index]
        mrl = encoded_data[1][index]

        training[0][tr_count] = data
        training[1][tr_count] = mrl

        tr_count += 1

    return training, index_list

def assign_validating(encoded_data: tuple[np.ndarray, np.ndarray], index_list: list[float], num_seq: int) -> tuple[tuple[np.ndarray, np.ndarray], list[int]]:

    validating_fract = 0.1
    validating_num = int(num_seq * validating_fract)

    seed = 42
    random_num_gen = random.Random(seed)

    validating_indexes = []

    validating = (np.zeros((validating_num, 4, 50)), np.zeros(validating_num))

    for i in range(validating_num):

        index_list_len = len(index_list)

        random_num = random_num_gen.randint(0, (index_list_len - 1))

        random_index = index_list.pop(random_num)

        validating_indexes.append(random_index)

        if i == validating_num - 1:
            print(i+1, "Validating indexes found.")

        va_count = 0

    for index in validating_indexes:

        data = encoded_data[0][index]
        mrl = encoded_data[1][index]

        validating[0][va_count] = data
        validating[1][va_count] = mrl

        va_count += 1

    return validating, index_list

def assign_testing(encoded_data: tuple[np.ndarray, np.ndarray], index_list: list[float]) -> tuple[tuple[np.ndarray, np.ndarray], list[int]]:

    testing_num = len(index_list)

    print(testing_num, "Testing indexes found.")

    testing = (np.zeros((testing_num, 4, 50)), np.zeros(testing_num))


    va_count = 0

    for index in index_list:

        data = encoded_data[0][index]
        mrl = encoded_data[1][index]

        testing[0][va_count] = data
        testing[1][va_count] = mrl

        va_count += 1

    return testing



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
