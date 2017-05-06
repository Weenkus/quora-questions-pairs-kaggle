import csv
from math import log
import spacy
import pickle
from collections import Counter

nlp = spacy.load('en_default')


def all_questions():
    with open('input/train.csv', 'r') as input_file:
        reader = csv.DictReader(input_file)
        for line in reader:
            yield line['question1']
            yield line['question2']

    with open('input/test.csv', 'r') as input_file:
        reader = csv.DictReader(input_file)
        for line in reader:
            yield line['question1']
            yield line['question2']


def extract_lemmas(question):
    document = nlp(question)
    bigrams = set()
    previous_word = None
    for current_word in document:
        if previous_word is None:
            continue
        bigram = (previous_word.lemma_, current_word.lemma_)
        bigrams.add(bigram)
    for bigram in bigrams:
        yield bigram


used_questions = set()
bigram_counter = Counter()

for question in all_questions():
    if question not in used_questions:
        used_questions.add(question)
        for bigram in extract_lemmas(question):
            bigram_counter.update([bigram])

with open('input/bigram_frequencies_quora.pickle', 'wb') as output_file:
    pickle.dump(bigram_counter, output_file)

idfs = {lemma: log(len(used_questions)/count) for lemma, count in bigram_counter.items()}

with open('input/idfs_bigram.pickle', 'wb') as output_file:
    pickle.dump(idfs, output_file)
