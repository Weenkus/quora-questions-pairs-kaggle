import spacy
from spacy.symbols import nsubj, dobj

nlp = spacy.load('en_default')

interrogative_words = {'what', 'why', 'who', 'when', 'whom', 'how', 'where', 'which', 'whose', 'be', 'can', 'do'}


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
