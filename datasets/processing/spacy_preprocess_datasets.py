import spacy
import pickle
import csv

nlp = spacy.load('en_default')


def read_dataset(file_path):
    with open(file_path, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for line in reader:
            yield line


def process_line(line_dict):
    data = {
        'id': int(line_dict['test_id']) if 'test_id' in line_dict else int(line_dict['id']),
        'question1': nlp(line_dict['question1']).to_bytes(),
        'question2': nlp(line_dict['question2']).to_bytes()
    }
    if 'is_duplicate' in line_dict:
        data['is_duplicate'] = True if line_dict['is_duplicate'] == '1' else False
    return data


def process_dataset(dataset):
    return [process_line(line_dict) for line_dict in dataset]


with open('../input/train_dataset.pickle', 'wb') as output_file:
   pickle.dump(process_dataset(read_dataset('../input/train.csv')), output_file)

with open('../input/test_dataset.pickle', 'wb') as output_file:
    pickle.dump(process_dataset(read_dataset('../input/test.csv')), output_file)
