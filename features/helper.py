import math
import pickle
import spacy
from spacy.symbols import nsubj, dobj
from scipy.stats.mstats import gmean

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
    return gmean(list(map(get_unigram_idf, document)))
