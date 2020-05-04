import os
from typing import List

import bert
import pandas as pd
import tensorflow as tf

from config import BERT_DIR, MAX_LEN, CLASSES, ensure_bert_data

VOCAB = bert.bert_tokenization.load_vocab(os.path.join(BERT_DIR, 'vocab.txt'))
WORD_TOKENIZER = bert.bert_tokenization.WordpieceTokenizer(VOCAB)

ABBREVIATIONS = u"""
ім
о
вул
просп
бул
пров
пл
г
р
див
п
с
м
тис
""".strip().split()

DELIMITERS = ['.', '!', '?', ' ', ',', ':', ';']


def tokenize(text: str, ann: pd.DataFrame = None) -> (List[List[str]], List[List[str]]):
    inputs = []
    labels = []

    ann_index = 0
    ch_index = 0

    token = ''
    label = 'NONE'

    close_context_ch = None

    sub_inputs = []
    sub_labels = []

    def append_token():
        if len(token) > 0 and not token.isspace():
            sub_tokens = WORD_TOKENIZER.tokenize(token)
            for sub_token in sub_tokens:
                sub_inputs.append(sub_token)
                if ann is not None:
                    sub_labels.append(label)

    def append_sub_input():
        if len(sub_inputs) > 0:
            inputs.append(sub_inputs)
        if len(sub_labels) > 0:
            labels.append(sub_labels)

    def is_skip_dot():
        if token in ABBREVIATIONS:
            return True
        if len(token) == 1:
            return True
        if token.isdigit() \
                and ch_index < len(text) - 1 \
                and text[ch_index + 1].isdigit():
            return True
        return False

    while ch_index < len(text):
        ch = text[ch_index]

        if ch == '\n' or ch == '\r':
            if len(sub_inputs) > 0:
                append_token()

                token = ''

                append_sub_input()
                sub_inputs = []
                sub_labels = []

            if ann is not None and ann_index < ann.shape[0]:
                if ch_index >= ann.iloc[ann_index]['End']:
                    ann_index += 1
                    label = 'NONE'

            close_context_ch = None
        else:
            if close_context_ch is None and ch in DELIMITERS:
                if ch == '.' and is_skip_dot():
                    token += ch
                else:
                    append_token()

                    if ann is not None and ann_index < ann.shape[0]:
                        if ch_index >= ann.iloc[ann_index]['End']:
                            ann_index += 1
                            label = 'NONE'

                    if (ch == '.' or ch == '?' or ch == '!') \
                            and ch_index < len(text) - 2 \
                            and text[ch_index + 1] == '.' \
                            and text[ch_index + 2] == '.':
                        if ch_index < len(text) - 3 \
                                and text[ch_index + 3] == ')':
                            token = '...)'
                            ch_index += 3
                        else:
                            token = '...'
                            ch_index += 2
                    elif ch == '?' and ch_index < len(text) - 1 \
                            and text[ch_index + 1] == '!':
                        token = '?!'
                        ch_index += 1
                    else:
                        token = ch

                    append_token()

                    if ch in ['.', '!', '?']:
                        append_sub_input()
                        sub_inputs = []
                        sub_labels = []

                    token = ''
            else:
                if close_context_ch is None:
                    if ch == '"':
                        close_context_ch = '"'
                    elif ch == '(':
                        close_context_ch = ')'
                    elif ch == '«':
                        close_context_ch = '»'
                elif ch == close_context_ch:
                    close_context_ch = None

                if not ch.isalnum():
                    append_token()

                    token = ch
                    if ann is not None and ann_index < ann.shape[0]:
                        if ch_index >= ann.iloc[ann_index]['End']:
                            ann_index += 1
                            label = 'NONE'

                    if ann is not None and ann_index < ann.shape[0]:
                        if ann_index < ann.shape[0] and ann.iloc[ann_index]['Start'] <= ch_index:
                            label = ann.iloc[ann_index]['Type']

                    append_token()

                    token = ''
                else:
                    if ann is not None and ann_index < ann.shape[0]:
                        if ann_index < ann.shape[0] and ann.iloc[ann_index]['Start'] <= ch_index:
                            label = ann.iloc[ann_index]['Type']

                    token += ch

        ch_index += 1

    append_token()
    append_sub_input()

    return inputs, labels


def encode_inputs(inputs: List[List[str]], remove_too_long: bool = True) -> List[List[int]]:
    encoded_inputs = list(map(lambda t: [VOCAB['[CLS]']] + list(map(lambda tt: VOCAB[tt], t)) + [VOCAB['[SEP]']],
                              inputs))
    if remove_too_long:
        encoded_inputs = list(filter(lambda i: len(i) <= MAX_LEN, encoded_inputs))
    encoded_inputs = tf.keras.preprocessing.sequence.pad_sequences(encoded_inputs,
                                                                   maxlen=MAX_LEN,
                                                                   padding='post',
                                                                   truncating='post',
                                                                   value=0)
    return encoded_inputs


def encode_labels(labels: List[List[str]], remove_too_long: bool = True) -> List[List[int]]:
    encoded_labels = list(map(lambda t: [0] + list(map(lambda tt: CLASSES.index(tt), t)) + [0],
                              labels))
    if remove_too_long:
        encoded_labels = list(filter(lambda i: len(i) <= MAX_LEN, encoded_labels))
    encoded_labels = tf.keras.preprocessing.sequence.pad_sequences(encoded_labels,
                                                                   maxlen=MAX_LEN,
                                                                   padding='post',
                                                                   truncating='post',
                                                                   value=0)
    encoded_labels = tf.keras.utils.to_categorical(encoded_labels, len(CLASSES))
    return encoded_labels
