import spacy
from spacy.tokens.doc import Doc
import pickle


def deserialize_dataset(file_path, max_items):
    vocab = spacy.load('en_default').vocab

    with open(file_path, 'rb') as input_file:
        for data_point in pickle.load(input_file)[:max_items]:
            yield {
                'question1': Doc(vocab).from_bytes(data_point['question1']),
                'question2': Doc(vocab).from_bytes(data_point['question2']),
                'id': data_point['id'],
                'is_duplicate': data_point.get('is_duplicate', None)
            }


def test_set(max_items=2345805):
    return deserialize_dataset('input/test_dataset.pickle', max_items)


def train_set(max_items=404301):
    return deserialize_dataset('input/train_dataset.pickle', max_items)
