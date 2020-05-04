import os
import sys
from typing import NoReturn

from config import CHECKPOINT_FILE_NAME, BATCH_SIZE, CLASSES
from models import create_model
from preprocessing import tokenize, encode_inputs


def main() -> NoReturn:
    with open(sys.argv[1], 'r') as f:
        text = f.read()

    inputs, _ = tokenize(text)
    encoded_inputs = encode_inputs(inputs, remove_too_long=False)

    model = create_model()

    checkpoint_file_path = os.environ.get('CHECKPOINT_FILE',
                                          os.path.join(os.environ.get('CHECKPOINTS_DIR', '.'),
                                                       CHECKPOINT_FILE_NAME))
    model.load_weights(checkpoint_file_path)
    print(f'Weights is loaded from: {checkpoint_file_path}')

    classes = model.predict_classes(encoded_inputs,
                                    batch_size=BATCH_SIZE,
                                    verbose=1)

    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            token = inputs[i][j]
            if j < len(classes[i]) - 1:
                cl = classes[i][j + 1]
            else:
                cl = 0

            if token.startswith('##'):
                print(token[2:], end='')
            else:
                if j > 0:
                    print(' ', end='')
                if cl > 0:
                    print(f'[{CLASSES[cl]}] ', end='')
                print(token, end='')
        print('\n', end='')


if __name__ == '__main__':
    main()
