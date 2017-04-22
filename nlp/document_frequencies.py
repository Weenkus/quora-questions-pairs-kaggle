from collections import Counter
from pipeline.pipey import modifier, apply_pipeline
import spacy
import pickle

nlp = spacy.load('en_default')


def all_questions():
    with open('../input/all_questions_corpus.txt', 'r', encoding="ISO-8859-1") as input_file:
        for line in input_file:
            yield line.strip()


def get_lemma_set(question_document):
    return set([(word.lemma_, word.pos_) for word in question_document if not word.is_stop and not word.is_punct])


word_counter = Counter()


pipeline = [
    (nlp, modifier.map),
    (get_lemma_set, modifier.map),
]

for word_set in apply_pipeline(all_questions(), pipeline):
    word_counter.update(word_set)

print(len(word_counter))
print(word_counter.most_common(20))

with open('../input/document_frequencies.pickle', 'wb') as output_file:
    pickle.dump(word_counter, output_file)
