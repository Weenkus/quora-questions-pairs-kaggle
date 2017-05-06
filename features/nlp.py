from .helper import jaccard_index, nlp, get_heads, get_objects, get_roots, get_subjects, interrogative_words, \
    get_non_alphanumeric_characters, filter_words_with_minimum_idf, geometric_mean_of_unigram_idfs


def assert_valid_input(entry):
    assert isinstance(entry, dict)
    assert 'question1' in entry
    assert 'question2' in entry
    assert isinstance(entry['question1'], str)
    assert isinstance(entry['question2'], str)


def spacy_process(entry):
    entry['question1_document'] = nlp(entry['question1'])
    entry['question2_document'] = nlp(entry['question2'])


def simple_similarity(entry):
    entry['spacy_similarity_feature'] = entry['question1_document'].similarity(entry['question2_document'])


def entity_sets_similarity(entry):
    entry['entities_similarity_feature'] = jaccard_index(
        set(entry['question1_document'].ents),
        set(entry['question2_document'].ents)
    )


def numbers_sets_similarity(entry):
    entry['numbers_similarity_feature'] = jaccard_index(
        set([word.lemma for word in entry['question1_document'] if word.like_num]),
        set([word.lemma for word in entry['question2_document'] if word.like_num])
    )


def subject_sets_similarity(entry):
    entry['subjects_similarity_feature'] = jaccard_index(
        get_subjects(entry['question1_document']),
        get_subjects(entry['question2_document'])
    )


def parse_roots_sets_similarity(entry):
    entry['roots_similarity_feature'] = jaccard_index(
        get_roots(entry['question1_document']),
        get_roots(entry['question2_document'])
    )


def parse_heads_sets_similarity(entry):
    entry['heads_similarity_feature'] = jaccard_index(
        get_heads(entry['question1_document']),
        get_heads(entry['question2_document'])
    )


def object_sets_similarity(entry):
    entry['objects_similarity_feature'] = jaccard_index(
        get_objects(entry['question1_document']),
        get_objects(entry['question2_document'])
    )


def first_interrogative_matching(entry):
    interrogatives1 = [word.lemma for word in entry['question1_document'] if word.lemma_ in interrogative_words]
    interrogatives2 = [word.lemma for word in entry['question2_document'] if word.lemma_ in interrogative_words]

    match = False
    if interrogatives1 and interrogatives2:
        match = interrogatives1[0] == interrogatives2[0]
    elif not interrogatives1 and not interrogatives2:
        match = True

    entry['interrogative_match_feature'] = float(match)


def non_alphanumeric_sets_similarity(entry):
    entry['non_alphanumeric_similarity_feature'] = jaccard_index(
        set(get_non_alphanumeric_characters(entry['question1'])),
        set(get_non_alphanumeric_characters(entry['question2']))
    )


def unigram_idf_cutoff_similarity(entry):
    entry['unigram_idf_cutoff_similarity_5_feature'] = jaccard_index(
        filter_words_with_minimum_idf(entry['question1_document'], 5),
        filter_words_with_minimum_idf(entry['question2_document'], 5)
    )
    entry['unigram_idf_cutoff_similarity_10_feature'] = jaccard_index(
        filter_words_with_minimum_idf(entry['question1_document'], 10),
        filter_words_with_minimum_idf(entry['question2_document'], 10)
    )
    entry['unigram_idf_cutoff_similarity_15_feature'] = jaccard_index(
        filter_words_with_minimum_idf(entry['question1_document'], 15),
        filter_words_with_minimum_idf(entry['question2_document'], 15)
    )


def unigram_idf_mean_difference(entry):
    entry['unigram_idf_mean_difference_feature'] = abs(
        geometric_mean_of_unigram_idfs(entry['question1_document']) -
        geometric_mean_of_unigram_idfs(entry['question2_document'])
    )

