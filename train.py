import os
from typing import List, NoReturn

import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import CHECKPOINT_FILE_NAME, BATCH_SIZE
from models import create_model
from preprocessing import encode_inputs, encode_labels, tokenize


def prepare_train_data() -> (List[List[str]], List[List[str]]):
    import pandas as pd

    inputs = []
    labels = []

    if not os.path.isdir('ner-uk'):
        import subprocess

        subprocess.run(['git',
                        'clone',
                        'https://github.com/lang-uk/ner-uk'])

    for root, _, files in os.walk('ner-uk/data/'):
        for file in files:
            path = os.path.join(root, file)
            try:
                path_without_ext, ext = os.path.splitext(path)
                ann_path = path_without_ext + '.ann'

                if os.path.isfile(path) \
                        and ext == '.txt' and not path.endswith('.tok.txt') \
                        and os.path.isfile(ann_path):
                    with open(path, 'r') as f:
                        content = f.read()

                    ann = pd.read_csv(ann_path,
                                      sep='\t',
                                      header=None,
                                      names=['Index', 'Type', 'Snippet'])
                    ann[['Type', 'Start', 'End']] = ann['Type'].str.split(' ', expand=True)
                    ann[['Start', 'End']] = ann[['Start', 'End']].astype(int)

                    sub_inputs, sub_labels = tokenize(content, ann)

                    inputs.extend(sub_inputs)
                    labels.extend(sub_labels)
            except Exception as ex:
                print(f'{path} -> {ex}')

    return inputs, labels


def load_train_data() -> (List[List[str]], List[List[str]]):
    import pickle

    inputs_cache_path = 'train-inputs.pickle'
    labels_cache_path = 'train-labels.pickle'

    if not os.path.isfile(inputs_cache_path) or not os.path.isfile(labels_cache_path):
        inputs, labels = prepare_train_data()

        with open(inputs_cache_path, 'wb') as f:
            pickle.dump(inputs, f)
        with open(labels_cache_path, 'wb') as f:
            pickle.dump(labels, f)
    else:
        with open(inputs_cache_path, 'rb') as f:
            inputs = pickle.load(f)
        with open(labels_cache_path, 'rb') as f:
            labels = pickle.load(f)

    return inputs, labels


def main() -> NoReturn:
    checkpoint_file_path = os.path.join(os.environ.get('CHECKPOINTS_DIR', '.'), CHECKPOINT_FILE_NAME)
    print(f'Checkpoint file: {checkpoint_file_path}')

    inputs, labels = load_train_data()

    for i in reversed(range(len(labels))):
        any_with_class = False
        for j in range(len(labels[i])):
            if labels[i][j] != 'NONE':
                any_with_class = True
                break
        if not any_with_class:
            del inputs[i]
            del labels[i]

    encoded_inputs = encode_inputs(inputs)
    encoded_labels = encode_labels(labels)

    x_train, x_test, y_train, y_test = train_test_split(encoded_inputs,
                                                        encoded_labels,
                                                        test_size=0.15,
                                                        random_state=17)
    print(f'Loaded {len(encoded_inputs)} samples')

    model = create_model()

    if os.path.isfile(checkpoint_file_path):
        model.load_weights(checkpoint_file_path)
        print(f'Weights is loaded from: {checkpoint_file_path}')

    print(model.summary())

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_file_path,
                                                    verbose=1,
                                                    save_best_only=False,
                                                    save_weights_only=True,
                                                    save_freq=4*BATCH_SIZE)

    def lr_scheduler(epoch):
        lr = 5e-5 * round(0.1 ** epoch, 10)
        return lr

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    callbacks = [checkpoint, lr_scheduler]

    model.fit(x_train,
              y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              epochs=3,
              initial_epoch=int(os.environ.get('INITIAL_EPOCH', '0')),
              verbose=1)

    # model.evaluate(x_test,
    #                y_test,
    #                batch_size=BATCH_SIZE,
    #                verbose=1)


if __name__ == '__main__':
    main()
