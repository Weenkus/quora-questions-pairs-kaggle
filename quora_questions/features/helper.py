import math
import pickle
import spacy
from spacy.symbols import nsubj, dobj, VERB
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
    return _levenshtein_distance(s1, s2) / maximum_length


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
