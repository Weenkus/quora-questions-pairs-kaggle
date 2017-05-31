import gc
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

features = [
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


def load_train():
    train_old = pd.read_pickle('../../features/train_new.pkl')
    train_bekavac = pd.read_csv('../../features/train_features_bekavac.csv')
    train_magic1 = pd.read_csv('../../features/train_magic_feature_v1.csv')
    train_magic2 = pd.read_csv('../../features/train_magic_feature_v2.csv')
    train_magic3 = pd.read_csv('../../features/train_magic_feature_v3.csv')
    abhishek_train = pd.read_csv('../../features/abhishek_train_features.csv', encoding="ISO-8859-1")

    train_magic1 = train_magic1.drop('is_duplicate', 1)
    train_magic1 = train_magic1.drop('question2', 1)
    train_magic1 = train_magic1.drop('question1', 1)

    train_magic3 = train_magic3.drop('is_duplicate', 1)
    train_magic3 = train_magic3.drop('question2', 1)
    train_magic3 = train_magic3.drop('question1', 1)

    abhishek_train = abhishek_train.drop('question2', 1)
    abhishek_train = abhishek_train.drop('question1', 1)

    train = pd.concat([train_old, train_bekavac, train_magic1, train_magic2, train_magic3, abhishek_train], axis=1)

    del train_old, train_bekavac, train_magic1, train_magic2
    gc.collect()

    return train


def load_test():
    test_old = pd.read_pickle('../../features/test_new.pkl')
    test_bekavac = pd.read_csv('../../features/test_features_bekavac.csv')
    test_magic1 = pd.read_csv('../../features/test_magic_feature_v1.csv')
    test_magic3 = pd.read_csv('../../features/test_magic_feature_v3.csv')
    abhishek_test = pd.read_csv('../../features/abhishek_test_features.csv', encoding="ISO-8859-1")

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


def get_model():
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.65,
        gamma=1.5,
        seed=42,
        colsample_bytree=0.3
    )

    return model


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


    print('Training')
    model = get_model()
    model.fit(X_train, y_train)

    del train
    gc.collect()

    print('Loading test')
    test = load_test()
    X_test = test[features]
    
    del test
    gc.collect()

    print('Generating predictions without correction')
    predictions = model.predict_proba(X_test)
    generate_submission(predictions, 'xgb5_yO_nC')

    print('Generating predictions with correction')
    predictions = prediction_correction(predictions)
    generate_submission(predictions, 'xgb5_yO_yC')

if __name__ == '__main__':
    main()
