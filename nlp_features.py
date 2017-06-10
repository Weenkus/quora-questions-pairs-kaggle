import csv
import datetime

from quora_questions.features.nlp import assert_valid_input, spacy_process, simple_similarity, entity_sets_similarity, \
    numbers_sets_similarity, subject_sets_similarity, parse_roots_sets_similarity, parse_heads_sets_similarity, \
    object_sets_similarity, first_interrogative_matching, non_alphanumeric_sets_similarity, \
    unigram_idf_cutoff_similarity, unigram_idf_mean_difference, subject_verb_inversion_similarity, \
    number_of_children_similarity, document_pos_cutoff_similarity, compression_size_reduction_ratio, \
    email_sets_similarity, filtered_cosine_similarity, url_sets_similarity, first_word_similarity, \
    last_word_similarity, lemma_edit_distance, question_length_similarity
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
    (filtered_cosine_similarity, modifier.map),
    (entity_sets_similarity, modifier.map),
    (numbers_sets_similarity, modifier.map),
    (email_sets_similarity, modifier.map),
    (url_sets_similarity, modifier.map),
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
    (document_pos_cutoff_similarity, modifier.map),
    (compression_size_reduction_ratio, modifier.map),
    (first_word_similarity, modifier.map),
    (last_word_similarity, modifier.map),
    (lemma_edit_distance, modifier.map),
    (question_length_similarity, modifier.map)
]


def create_features(input_file_path):
    for entry in apply_pipeline(read_dataset(input_file_path), nlp_pipeline):
        yield {k: v for k, v in entry.items() if k.endswith('feature') or k == 'id'}


def write_results_to_csv(results, output_file_path):
    try:
        first_row = next(results)
    except StopIteration:
        return

    with open(output_file_path) as output_file:
        field_names = list(first_row.keys())
        csv_writer = csv.DictWriter(output_file, fieldnames=field_names)
        csv_writer.writeheader()
        field_names_set = set(field_names)

        for row in results:
            assert set(row.keys()) == field_names_set
            csv_writer.writerow(row)


print(datetime.datetime.now())
write_results_to_csv(create_features('input/train.csv'), 'output/train_features.csv')

print(datetime.datetime.now())
write_results_to_csv(create_features('input/test.csv'), 'output/test_features.csv')

print(datetime.datetime.now())
