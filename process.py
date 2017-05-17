from quora_questions.features.features_pipeline import apply_pipeline, read_dataset, nlp_pipeline

for entry in apply_pipeline(read_dataset('input/train.csv'), nlp_pipeline):
    print(entry)
