import bert
import tensorflow as tf

from tensorflow import keras as k

from config import BERT_DIR, BERT_WEIGHTS_PATH, MAX_LEN, CLASSES


def create_bert_layer() -> bert.BertModelLayer:
    bert_params = bert.params_from_pretrained_ckpt(BERT_DIR)
    bert_params.mask_zero = True

    bert_layer = bert.BertModelLayer.from_params(bert_params, name='bert')

    bert_layer.apply_adapter_freeze()

    return bert_layer


def create_model() -> k.Sequential:
    bert_layer = create_bert_layer()

    model = k.Sequential([
        k.layers.Input(shape=(MAX_LEN,), dtype='int32', name='input_ids'),
        bert_layer,
        k.layers.TimeDistributed(k.layers.Dense(768 * 3, activation=tf.nn.relu)),
        k.layers.TimeDistributed(k.layers.Dense(len(CLASSES), activation=tf.nn.softmax))
    ])

    model.build()

    bert_layer.apply_adapter_freeze()
    bert.load_stock_weights(bert_layer, BERT_WEIGHTS_PATH)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                  metrics=['categorical_accuracy'])

    return model
