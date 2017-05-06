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
    words = set()
    for word in document:
        words.add((word.lemma_, word.pos_))
    for word_pair in words:
        yield word_pair

used_questions = set()
lemma_counter = Counter()

for question in all_questions():
    if question not in used_questions:
        used_questions.add(question)
        for lemma in extract_lemmas(question):
            lemma_counter.update([lemma])

with open('input/lemma_frequencies_quora.pickle', 'wb') as output_file:
    pickle.dump(lemma_counter, output_file)

idfs = {lemma: log(len(used_questions)/count) for lemma, count in lemma_counter.items()}

with open('input/idfs.pickle', 'wb') as output_file:
    pickle.dump(idfs, output_file)
