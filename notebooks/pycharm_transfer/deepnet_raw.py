
# coding: utf-8

# ## Libraries

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import configparser
import gc


# ## Load features
# To load the features you first have to create them, run the notebook feature_engineering. Beware it takes about 2-3 hours to run so save your features!

# In[2]:

train_old = pd.read_pickle('../../features/train_new.pkl')
print(train_old.shape)


# In[3]:

train_old = pd.read_pickle('../../features/train_new.pkl')
train_bekavac = pd.read_csv('../../features/train_features_bekavac_v2.csv')
train_magic1 = pd.read_csv('../../features/train_magic_feature_v1.csv')
train_magic2 = pd.read_csv('../../features/train_magic_feature_v2.csv')
train_magic3 = pd.read_csv('../../features/train_magic_feature_v3.csv')
abhishek_train = pd.read_csv('../../features/abhishek_train_features.csv', encoding="ISO-8859-1")
#abhishek_train = abhishek_train.fillna(abhishek_train.mean())

train_magic1 = train_magic1.drop('is_duplicate', 1)
train_magic1 = train_magic1.drop('question2', 1)
train_magic1 = train_magic1.drop('question1', 1)

train_magic3 = train_magic3.drop('is_duplicate', 1)
train_magic3 = train_magic3.drop('question2', 1)
train_magic3 = train_magic3.drop('question1', 1)

abhishek_train = abhishek_train.drop('question2', 1)
abhishek_train = abhishek_train.drop('question1', 1)

train = pd.concat([train_old, train_bekavac, train_magic1, train_magic2, train_magic3, abhishek_train], axis=1)

del train_old, train_bekavac, train_magic1, train_magic2, train_magic3, abhishek_train
gc.collect()



# ## Consts
# Always use constant SEED otherwise the experiment is not reproducable, in that case why are we doing it? 

# In[4]:

SEED = 42
NUM_WORDS = 70000
SEQ_MAX_LEN = 30
EMBEDDING_DIM = 300 # 50, 100, 200 or 300
P_RATE=0.174

np.random.seed(SEED)
tf.set_random_seed(SEED)


# ## Feature selection

# In[5]:

from sklearn.feature_selection import VarianceThreshold
from keras.preprocessing import sequence, text

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split


# In[6]:

features = list(set([
	'compression_ratio_feature', 'document_pos_similarity_10_feature',
       'document_pos_similarity_3_feature',
       'document_pos_similarity_5_feature',
       'document_pos_similarity_7_feature',
       'document_pos_similarity_all_feature', 'email_similarity_feature',
       'entities_similarity_feature', 'filtered_cosine_similarity_feature',
       'first_word_similarity_feature', 'heads_similarity_feature',
       'interrogative_match_feature', 'last_word_similarity_feature',
       'lemma_edit_distance_feature', 'non_alphanumeric_similarity_feature',
       'number_of_children_similarity_5_feature', 'numbers_similarity_feature',
       'objects_similarity_feature', 'question_length_similarity_feature',
       'roots_similarity_feature', 'spacy_similarity_feature',
       'subject_verb_inversion_similarity_feature',
       'subjects_similarity_feature',
       'unigram_idf_cutoff_similarity_10_feature',
       'unigram_idf_cutoff_similarity_12.5_feature',
       'unigram_idf_cutoff_similarity_15_feature',
       'unigram_idf_cutoff_similarity_1_feature',
       'unigram_idf_cutoff_similarity_5_feature',
       'unigram_idf_cutoff_similarity_7.5_feature',
       'unigram_idf_mean_difference_feature', 'url_similarity_feature',

    'q1_q2_wm_ratio', 'len_char_q1', 'len_char_q2', 'len_word_q1', 'len_word_q2',
    'common_words', 'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 'fuzz_partial_token_set_ratio',
    'fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio',
    'wmd', 'norm_wmd', 'cosine_distance', 'cityblock_distance', 'jaccard_distance',
    'canberra_distance', 'euclidean_distance', 'minkowski_distance', 'braycurtis_distance',
    'skew_q1vec', 'skew_q2vec', 'kur_q1vec', 'kur_q2vec',

    'q1_freq', 'q2_freq', 'q1_q2_intersect', 'word_share',
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
    'q1_SYM_count', 'q2_SYM_count', 'SYM_diff']))


target = 'is_duplicate'



# In[7]:

tokenizer = text.Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts( list(train.question1.values.astype(str)) + list(train.question2.values.astype(str)))

word_index = tokenizer.word_index
print('Found %d unique words in training set' % len(word_index))


# In[8]:

x1 = tokenizer.texts_to_sequences(train.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=SEQ_MAX_LEN)

x2 = tokenizer.texts_to_sequences(train.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=SEQ_MAX_LEN)


# In[9]:

X = train[features]
y = train[target]

X = np.array(X)
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)


# In[10]:

print(x1.shape, X.shape, y.shape)


# ## Cross validation

# In[11]:

X_train, X_val, x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
    X, x1, x2, y, test_size=0.1, random_state=SEED)

print(X_train.shape, x1_train.shape, y_train.shape)


# ## Pretrained embeddings
# Glove pretrained word2vec, source: https://nlp.stanford.edu/projects/glove/
# 
# Download: http://nlp.stanford.edu/data/glove.6B.zip
# 
# Use 300 dimensional vectors.

# In[12]:

import os


# In[13]:

embeddings_index = {}
with open('../../pretrained/glove.6B/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt', encoding='utf-8') as embedding_file:
    for line in embedding_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
print('Found %s pretrained word vectors.' % len(embeddings_index))


# Create a embedding matrix, each row coresponds to a token (id for a word) and contains a word2vec for that word.

# In[14]:

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
print(embedding_matrix.shape)


# ## Oversampling
# Oversampling leads to local validation score not matching the score from public LB on kaggle. Models with oversampling usually perform a bit better, but due to scores not maching if possible better not use it.
# 
# The idea for oversampling came from Kaggle (https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb) because the training and test set do not have the same distribution of dublicate questions. The train set has around 37% of duplicates while the private test set has 16.5% but the problem is that we only see the 35% of the prive test set. Final results are calculate on the remaining 65%, what if the distribution of the 35% set doe not match the other 65%, in that case oversampling while increasing the public LB score currently would yield in overfitting the score and poor results in the end.

# In[ ]:

def oversample(X, y, rate=P_RATE):
    pos_train = X[y == 1]
    neg_train = X[y == 0]

    # Now we oversample the negative class
    # There is likely a much more elegant way to do this...
    p = P_RATE
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -=1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print(len(pos_train) / (len(pos_train) + len(neg_train)))

    X = pd.concat([pos_train, neg_train])
    y = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

    return X, y

#y_untouched = y_train

#x1_train = pd.DataFrame(x1_train) 
#x2_train = pd.DataFrame(x2_train) 
#X_train = pd.DataFrame(X_train)

#X_train, y_train = oversample(X_train, y_untouched)
#x1_train, y_train = oversample(x1_train, y_untouched)
#x2_train, y_train = oversample(x2_train, y_untouched)

#X_train = np.array(X_train)
#x1_train = np.array(x1_train)
#x2_train = np.array(x2_train)


# ## Normalization
# Normalization helps but only if X is normalized, normalizing x1 and x2 does not allow the model to converge and pass the val_logloss of 0.42 -> bad. So far it seems that StandardScaler applied only on X does the trick.

# In[15]:

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[16]:

#scaler_X = StandardScaler()

#X_train = scaler_X.fit_transform(X_train)
#X_val = scaler_X.transform(X_val)


# In[ ]:

# scaler_x1 = MinMaxScaler()
# x1 = scaler_x1.fit_transform(x1)

# scaler_x2 = MinMaxScaler()
# x2 = scaler_x2.fit_transform(x2)


# ## Model

# In[17]:

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

from keras.models import load_model

import keras
from keras import backend as K
K.set_image_dim_ordering('tf')


# ### Create base models

# In[18]:

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
model_GRU.add(Merge([model_q1, model_q2], mode = 'concat'))
model_GRU.add(BatchNormalization())

model_GRU.add(Dense(512))
model_GRU.add(BatchNormalization())
model_GRU.add(Activation(elu))
model_GRU.add(Dropout(0.5))

#model_GRU.add(Dense(1, activation='sigmoid'))


# In[19]:

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
model_sum.add(Merge([model_sum1, model_sum2], mode = 'concat'))
model_sum.add(BatchNormalization())
model_sum.add(Dropout(0.15))

model_sum.add(Dense(512))
model_sum.add(BatchNormalization())
model_sum.add(Activation(elu))
model_sum.add(Dropout(0.4))

#model_sum.add(Dense(1, activation='sigmoid'))


# In[20]:

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
model_max.add(Merge([model_max1, model_max2], mode = 'concat'))
model_max.add(BatchNormalization())
model_max.add(Dropout(0.15))

model_max.add(Dense(512))
model_max.add(BatchNormalization())
model_max.add(Activation(elu))
model_max.add(Dropout(0.4))

#model_max.add(Dense(1, activation='sigmoid'))


# In[21]:

model_conv1 = Sequential()
model_conv1.add(Embedding(len(word_index) + 1,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=SEQ_MAX_LEN,
                     trainable=False,
                     dropout=0.15))

model_conv1.add(Convolution1D(filters = 256, kernel_size = 3, padding = 'same'))
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

model_conv2.add(Convolution1D(filters = 256, kernel_size = 3, padding = 'same'))
model_conv2.add(BatchNormalization())
model_conv2.add(Activation(elu))
model_conv2.add(Dropout(0.4))

model_conv2.add(Flatten())
model_conv2.add(Dense(256))
model_conv2.add(BatchNormalization())
model_conv2.add(Activation(elu))

model_glove_conv = Sequential()
model_glove_conv.add(Merge([model_conv1, model_conv2], mode = 'concat'))
model_glove_conv.add(BatchNormalization())

model_glove_conv.add(Dense(512))
model_glove_conv.add(BatchNormalization())
model_glove_conv.add(Activation(elu))
model_glove_conv.add(Dropout(0.5))

#model_glove_conv.add(Dense(1, activation='sigmoid'))


# In[22]:

model_features = Sequential()

model_features.add(Dense(256, input_dim=X_train.shape[1]))
model_features.add(BatchNormalization())
model_features.add(Activation(elu))

model_features.add(Dense(256,))
model_features.add(BatchNormalization())
model_features.add(Activation(elu))

model_features.add(Dense(512))
model_features.add(BatchNormalization())
model_features.add(Activation(elu))

model_features.add(Dense(512))
model_features.add(BatchNormalization())
model_features.add(Activation(elu))
model_features.add(Dropout(0.5))
                   
#model_features.add(Dense(1, activation='sigmoid'))


# In[23]:

merged_model = Sequential()
merged_model.add(Merge([model_GRU, model_sum, model_max, model_glove_conv, model_features], mode = 'concat'))
merged_model.add(BatchNormalization())
merged_model.add(Dropout(0.65))

merged_model.add(Dense(512))
merged_model.add(BatchNormalization())
merged_model.add(Activation(elu))
merged_model.add(Dropout(0.5))

merged_model.add(Dense(1, activation='sigmoid'))


train_data = [x1_train, x2_train, x1_train, x2_train, x1_train, x2_train, x1_train, x2_train, X_train]
val_data = [x1_val, x2_val, x1_val, x2_val, x1_val, x2_val, x1_val, x2_val, X_val]


print('Started training round 1')
merged_model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])


merged_model.fit(train_data, y_train,
          batch_size=16 * 10,
          epochs=3,
          verbose=0,
          validation_data=(val_data, y_val))

print('Started training round 2')
merged_model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])


merged_model.fit(train_data, y_train,
          batch_size=16 * 9,
          epochs=2,
          verbose=0,
          validation_data=(val_data, y_val))

print('Started training round 3')
merged_model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.00001),
              metrics=['accuracy'])


merged_model.fit(train_data, y_train,
          batch_size=16 * 8,
          epochs=1,
          verbose=0,
          validation_data=(val_data, y_val))


del x1_train, x2_train, X_train
del x1_val, x2_val, X_val
gc.collect()


test_old = pd.read_pickle('../../features/test_new.pkl')
test_bekavac = pd.read_csv('../../features/test_features_bekavac_v2.csv')
test_magic1 = pd.read_csv('../../features/test_magic_feature_v1.csv')
test_magic3 = pd.read_csv('../../features/test_magic_feature_v3.csv')
abhishek_test = pd.read_csv('../../features/abhishek_test_features.csv', encoding="ISO-8859-1")
#abhishek_test = abhishek_test.fillna(abhishek_test.mean())

test_magic1 = test_magic1.drop('is_duplicate', 1)
test_magic1 = test_magic1.drop('question2', 1)
test_magic1 = test_magic1.drop('question1', 1)

#test_magic3 = test_magic3.drop('is_duplicate', 1)
test_magic3 = test_magic3.drop('question2', 1)
test_magic3 = test_magic3.drop('question1', 1)

abhishek_test = abhishek_test.drop('question2', 1)
abhishek_test = abhishek_test.drop('question1', 1)

test_magic2 = pd.read_csv('../../features/test_magic_feature_v2.csv')

test = pd.concat([test_old, test_bekavac, test_magic1, test_magic2, test_magic3, abhishek_test], axis=1)

del test_old, test_bekavac, test_magic1, test_magic2, test_magic3, abhishek_test
gc.collect()


# ## Generate submission
# 
# Chunker is used to lower RAM requirements, without chunking requirement goes up to about 24GB of RAM.

# In[ ]:

import math

def chunker(collection, chunk_size=160000):
    chunk_num = math.ceil(collection.shape[0] / float(chunk_size))
    for i in range(chunk_num):
        yield collection[chunk_size*i : chunk_size*(i+1)]


# In[ ]:

preds = []
for q1, q2, test_row in zip(
    chunker(test.question1), chunker(test.question2), chunker(test)
):
    print('%d / %d' % (len(preds), len(test)))
    x1_test_row = tokenizer.texts_to_sequences(q1.values.astype(str))
    x1_test_row = sequence.pad_sequences(x1_test_row, maxlen=SEQ_MAX_LEN)

    x2_test_row = tokenizer.texts_to_sequences(q2.values.astype(str))
    x2_test_row = sequence.pad_sequences(x2_test_row, maxlen=SEQ_MAX_LEN)
    

    X_test_row = test_row[features]
    #X_test_row = scaler_X.transform(X_test_row)

    batch_preds = merged_model.predict([x1_test_row, x2_test_row, x1_test_row, x2_test_row, x1_test_row,
                                        x2_test_row, x1_test_row, x2_test_row, X_test_row],
                                       batch_size=16 * 6)

    preds.extend(batch_preds)



np.savetxt(
    '../../submissions/submission_d4_nO_nC.csv', np.c_[range(len(preds)), preds],
    delimiter=',', header='test_id,is_duplicate', comments='', fmt='%d,%f'
)



a = P_RATE / 0.37
b = (1 - P_RATE) / (1 - 0.37)

def fix_predictions_for_test_distribution(x):
    return a * x / (a * x + b * (1 - x))

preds = list(map(fix_predictions_for_test_distribution, preds))


np.savetxt(
    '../../submissions/submission_d4_nO_yC.csv', np.c_[range(len(preds)), preds],
    delimiter=',', header='test_id,is_duplicate', comments='', fmt='%d,%f'
)
