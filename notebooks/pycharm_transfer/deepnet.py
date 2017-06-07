import gc
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import log_loss, auc, f1_score

from keras.layers.advanced_activations import PReLU
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers import Merge

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, BatchNormalization, TimeDistributed, Input
from keras.layers import MaxPooling1D, Lambda, Convolution1D, Flatten, SpatialDropout1D
from keras_tqdm import TQDMNotebookCallback
from keras.layers.merge import Concatenate

from keras.optimizers import Adam, RMSprop, Adamax, Adagrad, Nadam
from keras.activations import elu, relu, tanh, sigmoid
from keras.preprocessing import sequence, text
from keras.models import load_model
import os
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

features = ['q1_freq', 'q2_freq', 'q1_q2_intersect',
            'word_share',
            'start_with_same_world', 'q1_char_num', 'q2_char_num', 'q1_word_num',
            'q2_word_num', 'rfidf_share', 'char_difference', 'word_difference',
            'seq_simhash_distance', 'shingle_simhash_distance', 'avg_word_len_q1',
            'avg_word_len_q2', 'avg_word_difference', 'unigrams_common_count',
            'bigrams_common_count', 'unigrams_common_ratio', 'bigrams_common_ratio',
            'cosin_sim', 'word2vec_q1_mean', 'word2vec_q2_mean', 'q1_NN_count',
            'q2_NN_count', 'NN_diff', 'q1_RB_count', 'q2_RB_count', 'RB_diff',
            'q1_VB_count', 'q2_VB_count', 'VB_diff', 'q1_DT_count', 'q2_DT_count',
            'DT_diff', 'q1_JJ_count', 'q2_JJ_count', 'JJ_diff', 'q1_FW_count',
            'q2_FW_count', 'FW_diff', 'q1_RP_count', 'q2_RP_count', 'RP_diff',
            'q1_SYM_count', 'q2_SYM_count', 'SYM_diff',
            'document_pos_similarity_10_feature',
            'document_pos_similarity_3_feature', 'entities_similarity_feature',
            'heads_similarity_feature', 'interrogative_match_feature',
            'non_alphanumeric_similarity_feature',
            'number_of_children_similarity_5_feature', 'numbers_similarity_feature',
            'objects_similarity_feature', 'roots_similarity_feature',
            'spacy_similarity_feature', 'subject_verb_inversion_similarity_feature',
            'subjects_similarity_feature',
            'unigram_idf_cutoff_similarity_10_feature',
            'unigram_idf_cutoff_similarity_15_feature',
            'unigram_idf_cutoff_similarity_5_feature',
            'unigram_idf_mean_difference_feature']
target = 'is_duplicate'

SEED = 42
UM_WORDS = 80000
SEQ_MAX_LEN = 25
EMBEDDING_DIM = 100  # 50, 100, 200 or 300

np.random.seed(SEED)
tf.set_random_seed(SEED)


def load_train():
    train_old = pd.read_pickle('../../features/train_new.pkl')
    train_bekavac = pd.read_csv('../../features/train_features_bekavac.csv')
    train_magic1 = pd.read_csv('../../features/train_magic_feature_v1.csv')
    train_magic2 = pd.read_csv('../../features/train_magic_feature_v2.csv')

    train_magic1 = train_magic1.drop('is_duplicate', 1)
    train_magic1 = train_magic1.drop('question2', 1)
    train_magic1 = train_magic1.drop('question1', 1)

    train = pd.concat([train_old, train_bekavac, train_magic1, train_magic2], axis=1)

    del train_old, train_bekavac, train_magic1, train_magic2
    gc.collect()

    return train


def load_test():
    test_old = pd.read_pickle('../../features/test_new.pkl')
    test_bekavac = pd.read_csv('../../features/test_features_bekavac.csv')
    test_magic1 = pd.read_csv('../../features/test_magic_feature_v1.csv')

    test_magic1 = test_magic1.drop('is_duplicate', 1)
    test_magic1 = test_magic1.drop('question2', 1)
    test_magic1 = test_magic1.drop('question1', 1)

    test_magic2 = pd.read_csv('../../features/test_magic_feature_v2.csv')

    test = pd.concat([test_old, test_bekavac, test_magic1, test_magic2], axis=1)

    del test_old, test_bekavac, test_magic1, test_magic2
    gc.collect()

    return test


def get_X_y(data):
    X = data[features]
    y = data[target]

    return X, y


def oversample(X, y, p=0.174):
    pos_train = X[y == 1]
    neg_train = X[y == 0]

    # Now we oversample the negative class
    # There is likely a much more elegant way to do this...
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

    X = pd.concat([pos_train, neg_train])
    y = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

    del pos_train, neg_train
    gc.collect()

    return X, y


def get_embedding_matrix(train):
    tokenizer = text.Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(list(train.question1.values.astype(str)) +
                           list(train.question2.values.astype(str)))

    word_index = tokenizer.word_index

    embeddings_index = {}
    with open('../../pretrained/glove.6B/glove.6B.' + str(EMBEDDING_DIM) +
                      'd.txt', encoding='utf-8') as embedding_file:

        for line in embedding_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s pretrained word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    x1 = tokenizer.texts_to_sequences(train.question1.values.astype(str))
    x1 = sequence.pad_sequences(x1, maxlen=SEQ_MAX_LEN)

    x2 = tokenizer.texts_to_sequences(train.question2.values.astype(str))
    x2 = sequence.pad_sequences(x2, maxlen=SEQ_MAX_LEN)

    x1 = np.array(x1)
    x2 = np.array(x2)

    return embedding_matrix, x1, x2, word_index


def get_model(word_index, embedding_matrix, X_train):
    model_q1 = Sequential()
    model_q1.add(Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights=[embedding_matrix],
                           input_length=SEQ_MAX_LEN,
                           trainable=False,
                           dropout=0.2))

    model_q1.add(GRU(256, recurrent_dropout=0.3, dropout=0.3, return_sequences=False))

    model_q2 = Sequential()
    model_q2.add(Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights=[embedding_matrix],
                           input_length=SEQ_MAX_LEN,
                           trainable=False,
                           dropout=0.2))

    model_q2.add(GRU(256, recurrent_dropout=0.3, dropout=0.3, return_sequences=False))

    model_GRU = Sequential()
    model_GRU.add(Merge([model_q1, model_q2], mode='concat'))
    model_GRU.add(BatchNormalization())

    model_GRU.add(Dense(512))
    model_GRU.add(BatchNormalization())
    model_GRU.add(Activation(elu))
    model_GRU.add(Dropout(0.5))

    model_sum1 = Sequential()
    model_sum1.add(Embedding(len(word_index) + 1,
                             EMBEDDING_DIM,
                             weights=[embedding_matrix],
                             input_length=SEQ_MAX_LEN,
                             trainable=False))

    model_sum1.add(TimeDistributed(Dense(300)))
    model_sum1.add(BatchNormalization())
    model_sum1.add(Activation(relu))

    model_sum1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

    model_sum2 = Sequential()
    model_sum2.add(Embedding(len(word_index) + 1,
                             EMBEDDING_DIM,
                             weights=[embedding_matrix],
                             input_length=SEQ_MAX_LEN,
                             trainable=False))

    model_sum2.add(TimeDistributed(Dense(300)))
    model_sum2.add(BatchNormalization())
    model_sum2.add(Activation(elu))

    model_sum2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

    model_sum = Sequential()
    model_sum.add(Merge([model_sum1, model_sum2], mode='concat'))
    model_sum.add(BatchNormalization())
    model_sum.add(Dropout(0.15))

    model_sum.add(Dense(512))
    model_sum.add(BatchNormalization())
    model_sum.add(Activation(elu))
    model_sum.add(Dropout(0.4))

    model_max1 = Sequential()
    model_max1.add(Embedding(len(word_index) + 1,
                             EMBEDDING_DIM,
                             weights=[embedding_matrix],
                             input_length=SEQ_MAX_LEN,
                             trainable=False))

    model_max1.add(TimeDistributed(Dense(300)))
    model_max1.add(BatchNormalization())
    model_max1.add(Activation(elu))

    model_max1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(300,)))

    model_max2 = Sequential()
    model_max2.add(Embedding(len(word_index) + 1,
                             EMBEDDING_DIM,
                             weights=[embedding_matrix],
                             input_length=SEQ_MAX_LEN,
                             trainable=False))

    model_max2.add(TimeDistributed(Dense(300)))
    model_max2.add(BatchNormalization())
    model_max2.add(Activation(elu))

    model_max2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(300,)))

    model_max = Sequential()
    model_max.add(Merge([model_max1, model_max2], mode='concat'))
    model_max.add(BatchNormalization())
    model_max.add(Dropout(0.15))

    model_max.add(Dense(512))
    model_max.add(BatchNormalization())
    model_max.add(Activation(elu))
    model_max.add(Dropout(0.4))

    model_conv1 = Sequential()
    model_conv1.add(Embedding(len(word_index) + 1,
                              EMBEDDING_DIM,
                              weights=[embedding_matrix],
                              input_length=SEQ_MAX_LEN,
                              trainable=False,
                              dropout=0.15))

    model_conv1.add(Convolution1D(filters=256, kernel_size=3, padding='same'))
    model_conv1.add(BatchNormalization())
    model_conv1.add(Activation(relu))
    model_conv1.add(Dropout(0.4))

    model_conv1.add(Flatten())
    model_conv1.add(Dense(256))
    model_conv1.add(BatchNormalization())
    model_conv1.add(Activation(relu))

    model_conv2 = Sequential()
    model_conv2.add(Embedding(len(word_index) + 1,
                              EMBEDDING_DIM,
                              weights=[embedding_matrix],
                              input_length=SEQ_MAX_LEN,
                              trainable=False,
                              dropout=0.15))

    model_conv2.add(Convolution1D(filters=256, kernel_size=3, padding='same'))
    model_conv2.add(BatchNormalization())
    model_conv2.add(Activation(elu))
    model_conv2.add(Dropout(0.4))

    model_conv2.add(Flatten())
    model_conv2.add(Dense(256))
    model_conv2.add(BatchNormalization())
    model_conv2.add(Activation(elu))

    model_glove_conv = Sequential()
    model_glove_conv.add(Merge([model_conv1, model_conv2], mode='concat'))
    model_glove_conv.add(BatchNormalization())

    model_glove_conv.add(Dense(512))
    model_glove_conv.add(BatchNormalization())
    model_glove_conv.add(Activation(elu))
    model_glove_conv.add(Dropout(0.5))

    model_features = Sequential()

    model_features.add(Dense(256, input_dim=X_train.shape[1]))
    model_features.add(BatchNormalization())
    model_features.add(Activation(elu))

    model_features.add(Dense(256, ))
    model_features.add(BatchNormalization())
    model_features.add(Activation(elu))

    model_features.add(Dense(512))
    model_features.add(BatchNormalization())
    model_features.add(Activation(elu))

    model_features.add(Dense(512))
    model_features.add(BatchNormalization())
    model_features.add(Activation(elu))
    model_features.add(Dropout(0.5))



    merged_model = Sequential()
    merged_model.add(Merge([model_GRU, model_sum, model_max, model_glove_conv, model_features], mode='concat'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dropout(0.35))

    merged_model.add(Dense(1024))
    merged_model.add(BatchNormalization())
    merged_model.add(Activation(elu))
    merged_model.add(Dropout(0.5))

    merged_model.add(Dense(1024))
    merged_model.add(BatchNormalization())
    merged_model.add(Activation(elu))
    merged_model.add(Dropout(0.5))

    merged_model.add(Dense(1, activation='sigmoid'))

    return merged_model


def generate_submission(predictions, name):
    np.savetxt(
        '../../submissions/' + name, np.c_[range(len(predictions)), predictions[:, 1]],
        delimiter=',', header='test_id,is_duplicate', comments='', fmt='%d,%f'
    )


def prediction_correction(predictions, p=0.174):
    a = p / 0.37
    b = (1 - p) / (1 - 0.37)

    def fix_predictions_for_test_distribution(x):
        return a * x / (a * x + b * (1 - x))

    predictions = np.array(list(map(fix_predictions_for_test_distribution, predictions)))
    return predictions


def main():
    print('Loading train')
    train = load_train()
    X_train, y_train = get_X_y(train)

    print('Oversampling')
    X_train, y_train = oversample(X_train, y_train)

    print('Creating an embedding matrix')
    embedding_matrix, x1, x2, word_index = get_embedding_matrix(X_train)

    print('Training')
    model = get_model(word_index, embedding_matrix, X_train)

    model.compile(loss='binary_crossentropy',
                         optimizer=RMSprop(0.001),
                         metrics=['accuracy'])

    X_train, X_val, x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
        X_train, x1, x2, y_train, test_size=0.1, random_state=SEED)

    train_data = [x1_train, x2_train, x1_train, x2_train, x1_train, x2_train, x1_train, x2_train, X_train]
    val_data = [x1_val, x2_val, x1_val, x2_val, x1_val, x2_val, x1_val, x2_val, X_val]


    model.fit(train_data, y_train,
                     batch_size=64 * 16,
                     epochs=4,
                     verbose=0,
                     validation_data=(val_data, y_val),
                     callbacks=[TQDMNotebookCallback()])


    print('Loading test')
    test = load_test()
    X_test = test[features]

    del test
    gc.collect()

    print('Generating predictions without correction')
    predictions = model.predict_proba(X_test)
    generate_submission(predictions, 'xgb3_yO_nC')

    print('Generating predictions with correction')
    predictions = prediction_correction(predictions)
    generate_submission(predictions, 'xgb3_yO_yC')


if __name__ == '__main__':
    main()
