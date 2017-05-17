import csv
import json

from quora_questions.features.nlp import assert_valid_input, spacy_process, simple_similarity, entity_sets_similarity, numbers_sets_similarity,\
    subject_sets_similarity, parse_roots_sets_similarity, parse_heads_sets_similarity, object_sets_similarity,\
    first_interrogative_matching, non_alphanumeric_sets_similarity, unigram_idf_cutoff_similarity,\
    unigram_idf_mean_difference, subject_verb_inversion_similarity, number_of_children_similarity,\
    document_pos_cutoff_similarity
from quora_questions.pipeline.pipey import apply_pipeline, modifier


def read_dataset(file_path):
    with open(file_path, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for line in reader:
            yield line


def process_line(line_dict):
    data = {
        'id': int(line_dict['test_id']) if 'test_id' in line_dict else int(line_dict['id']),
        'question1': line_dict['question1'],
        'question2': line_dict['question2']
    }
    if 'is_duplicate' in line_dict:
        data['is_duplicate'] = True if line_dict['is_duplicate'] == '1' else False
    return data

nlp_pipeline = [
    (process_line, modifier.map),
    (assert_valid_input, modifier.map),
    (spacy_process, modifier.map),
    (simple_similarity, modifier.map),
    (entity_sets_similarity, modifier.map),
    (numbers_sets_similarity, modifier.map),
    (subject_sets_similarity, modifier.map),
    (parse_roots_sets_similarity, modifier.map),
    (parse_heads_sets_similarity, modifier.map),
    (object_sets_similarity, modifier.map),
    (first_interrogative_matching, modifier.map),
    (non_alphanumeric_sets_similarity, modifier.map),
    (unigram_idf_cutoff_similarity, modifier.map),
    (unigram_idf_mean_difference, modifier.map),
    (subject_verb_inversion_similarity, modifier.map),
    (number_of_children_similarity, modifier.map),
    (document_pos_cutoff_similarity, modifier.map)
]

with open('output/train_features.json', 'w') as output_file:
    for entry in apply_pipeline(read_dataset('input/train.csv'), nlp_pipeline):
        filtered_entry = {k: v for k, v in entry.items() if k.endswith('feature') or k == 'id'}
        output_file.write('%s\n' % json.dumps(filtered_entry))

with open('output/test_features.json', 'w') as output_file:
    for entry in apply_pipeline(read_dataset('input/test.csv'), nlp_pipeline):
        filtered_entry = {k: v for k, v in entry.items() if k.endswith('feature') or k == 'id'}
        output_file.write('%s\n' % json.dumps(filtered_entry))
