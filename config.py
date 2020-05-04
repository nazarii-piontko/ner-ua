from os import path
from typing import NoReturn

BERT_DIR = path.join(path.dirname(path.realpath(__file__)), 'bert-model', 'multi_cased_L-12_H-768_A-12')
BERT_WEIGHTS_PATH = path.join(BERT_DIR, 'bert_model.ckpt')

CLASSES = ['NONE', 'PERS', 'LOC', 'ORG', 'MISC']

MAX_LEN = 256
BATCH_SIZE = 16

CHECKPOINT_FILE_NAME = 'model.hdf5'


def ensure_bert_data() -> NoReturn:
    if not path.isdir(BERT_DIR):
        import subprocess

        subprocess.run(['wget',
                        '-O',
                        'bert-model.zip',
                        'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'])
        subprocess.run(['unzip',
                        'bert-model.zip',
                        '-d',
                        'bert-model'])
        subprocess.run(['rm',
                        'bert-model.zip'])


ensure_bert_data()
