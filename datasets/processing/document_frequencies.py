from collections import Counter
from pipeline.pipey import modifier, apply_pipeline
import spacy
import pickle

nlp = spacy.load('en_default')


def all_questions():
    counter = 0
    with open('../input/all_questions_corpus.txt', 'r', encoding="ISO-8859-1") as input_file:
        for line in input_file:
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
            yield line.strip()


def get_lemma_set(question_document):
    return set([(word.lemma_, word.pos_) for word in question_document if not word.is_stop and not word.is_punct])


def update_counter(counter, iterable):
    counter.update(iterable)
    return counter


pipeline = [
    (nlp, modifier.map),
    (get_lemma_set, modifier.map),
    (update_counter, modifier.reduce, Counter())
]

word_counter = apply_pipeline(all_questions(), pipeline)

print(len(word_counter))
print(word_counter.most_common(20))

with open('../input/document_frequencies.pickle', 'wb') as output_file:
    pickle.dump(word_counter, output_file)
