import math
import pickle
import spacy
from spacy.symbols import nsubj, dobj, VERB
from scipy.stats.mstats import gmean
from scipy import spatial
import zlib
import sys

nlp = spacy.load('en_default')

interrogative_words = {'what', 'why', 'who', 'when', 'whom', 'how', 'where', 'which', 'whose', 'be', 'can', 'do'}


def _deserialize(file_path):
    with open(file_path, 'rb') as input_file:
        return pickle.load(input_file)

corpus_size = 5000000
unigram_idfs = _deserialize('input/idfs.pickle')


def jaccard_index(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def get_subjects(document):
    return set([word.lemma for word in document if word.dep == nsubj])


def get_objects(document):
    return set([word.lemma for word in document if word.dep == dobj])


def get_roots(document):
    return set([word.lemma for word in document if word.dep_ == 'ROOT'])


def get_heads(document):
    return set([word.head.lemma for word in document])


def get_non_alphanumeric_characters(text):
    return (character for character in text if not character.isalnum())


def get_unigram_idf(word):
    return unigram_idfs.get((word.lemma_, word.pos_), math.log(corpus_size))


def filter_words_with_minimum_idf(document, minimum_idf):
    return set(
        (word.lemma_ for word in document if get_unigram_idf(word) >= minimum_idf)
    )


def geometric_mean_of_unigram_idfs(document):
    document_idfs = list(map(get_unigram_idf, document))
    if len(document_idfs) == 0:
        return 0.0
    return gmean(document_idfs)


def _levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def relative_levenshtein_distance(s1, s2):
    maximum_length = max(len(s1), len(s2))
    try:
        return _levenshtein_distance(s1, s2) / maximum_length
    except ZeroDivisionError:
        return 1.0


def is_subject_verb_inversion(doc):
    verbs = set()
    for word in doc:
        if word.pos == VERB:
            verbs.add(word)
            continue
        if word.dep == nsubj and word.head.pos == VERB:
            return word.head in verbs
    return False


def naive_normalization(number):
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def number_of_children(document):
    return [len(list(word.children)) for word in document]


def document_pos(document):
    return [word.pos_ for word in document]


def _get_compressed_size(text):
    return sys.getsizeof(zlib.compress(bytes(text, 'utf-8'), 9))


def compare_compressed_size(text1, text2):
    size_sum = _get_compressed_size(text1) + _get_compressed_size(text2)
    size_joined = _get_compressed_size(text1 + text2)
    try:
        return size_joined / size_sum
    except ZeroDivisionError:
        return 1.0


def get_all_lemmas(document):
    return [word.lemma_ for word in document]


def simple_document_filter(document, use_out_of_vocabulary=False, use_stopwords=False, use_punctuation=False):
    return [
        word for word in document
        if (use_out_of_vocabulary or not word.is_oov)
        and (use_stopwords or not word.lemma_ in spacy.en.language_data.STOP_WORDS)
        and (use_punctuation or not word.is_punct)
        ]


def _get_average_word_vector(words):
    return sum([word.vector for word in words if word.has_vector]) / len([word for word in words if word.has_vector])


def get_cosine_similarity(words1, words2):
    try:
        return 1 - spatial.distance.cosine(_get_average_word_vector(words1), _get_average_word_vector(words2))
    except ZeroDivisionError:
        return 0.0


def relative_size_similarity(document1, document2):
    max_length = max(len(document1), len(document2))
    min_length = min(len(document1), len(document2))
    try:
        return min_length / max_length
    except ZeroDivisionError:
        return 0.0
