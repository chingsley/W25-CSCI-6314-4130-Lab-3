import numpy as np
import os

def _read_input(file, max_len=512):
    with open(file, 'r') as f:
        samples = []
        for line in f:
            # Split by tab
            ids = line.rstrip().split('\t')
            # Convert to ints
            ids = [int(i) for i in ids]
            # Truncate to max_len or pad with 0s
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [0] * (max_len - len(ids))
            samples.append(ids)
    
    return np.array(samples)


def process_train_and_test(data_dir, max_len=512):
    """
    Read the input data from the given directory and return the train and test data.
    """
    train_negative = _read_input(os.path.join(data_dir, 'train', '2d', 'Negative', 'tokenized1.tok'), max_len)
    train_positive = _read_input(os.path.join(data_dir, 'train', '2d', 'Positive', 'tokenized1.tok'), max_len)
    # Add the labels column
    train_positive_labels = np.ones(train_positive.shape[0]).reshape(-1, 1)
    train_negative_labels = np.zeros(train_negative.shape[0]).reshape(-1, 1)
    train_positive = np.hstack((train_positive, train_positive_labels))
    train_negative = np.hstack((train_negative, train_negative_labels))

    train = np.vstack((train_positive, train_negative))
    # Shuffle the data
    np.random.shuffle(train)
    

    test_negative = _read_input(os.path.join(data_dir, 'test', '2d', 'Negative', 'tokenized1.tok'), max_len)
    test_positive = _read_input(os.path.join(data_dir, 'test', '2d', 'Positive', 'tokenized1.tok'), max_len)
    # Add the labels column
    test_positive_labels = np.ones(test_positive.shape[0]).reshape(-1, 1)
    test_negative_labels = np.zeros(test_negative.shape[0]).reshape(-1, 1)
    test_positive = np.hstack((test_positive, test_positive_labels))
    test_negative = np.hstack((test_negative, test_negative_labels))

    test = np.vstack((test_positive, test_negative))
    # Shuffle the data
    np.random.shuffle(test)

    return train, test

def batchify(a, n=2):
    for i in range(a.shape[0] // n):
        yield a[n*i:n*(i+1)]

    

if __name__ == '__main__':
    train, test = process_train_and_test('/Users/mootez/courses/w25/6314/labs/lab3/W25-CSCI-6314-4130-Lab-3/data/train_test_dev_split_tokenized/UnutilizedAbstraction')
    train_batches = batchify(train, 16)
    for batch in train_batches:
        print(batch.shape)