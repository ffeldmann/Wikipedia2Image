""" Module containing basic data readers and extractors """

import json
import re
import pickle
import os
import unicodedata

def read_annotations(data_path):
    """
    read the json annotations (textual descriptions)
    :param data_path: path of the data folder, containing 'images' and 'texts' folders
    :return: annos => read annotations
    """

    annotations_filenames = os.listdir(data_path + '/text')
    images, descriptions = [], []  # initialize to empty lists

    for q in annotations_filenames:
        # save all image paths in a list
        q_number = q.split('.')[0]
        images.append(q_number + '.jpg')

        # and also corresponding descriptions
        with open(data_path + '/text/' + q, "r") as wiki_article_text:
            first_sentence = wiki_article_text.readlines()[0].split('.')[0]
        descriptions.append(first_sentence)


    # check if their lengths match:
    assert len(images) == len(descriptions), "something messed up while reading data ..."

    # return the read annos
    return images, descriptions # this is an image path and all the descriptions


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    import re
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def basic_preprocess(descriptions):
    """
    basic preprocessing on the input data
    :param descriptions: list[strings]
    :return: dat => list[lists[string]]
    """

    # insert space before all the special characters
    db_desc = []
    for desc in descriptions:
        desc = normalizeString(desc)
        punctuations = re.sub(r"([^a-zA-Z])", r" \1 ", desc)
        excess_space = re.sub('\s{2,}', ' ', punctuations)
        db_desc.append(excess_space)

    return db_desc


def frequency_count(text_data):
    """
    count the frequency of each word in data
    :param text_data: list[string]
    :return: freq_cnt => {word -> freq}
    """
    text_data = list(map(lambda x: x.split(), text_data))
    # generate the vocabulary
    total_word_list = []
    for line in text_data:
        total_word_list.extend(line)

    vocabulary = set(total_word_list)

    freq_count = dict(map(lambda x: (x, 0), vocabulary))

    # count the frequencies of the words
    for line in text_data:
        for word in line:
            freq_count[word] += 1

    # return the frequency counts
    return freq_count


def tokenize(text_data, freq_counts, vocab_size=None):
    """
    tokenize the text_data using the freq_counts
    :param text_data: list[string]
    :param freq_counts: {word -> freq}
    :param vocab_size: size of the truncated vocabulary
    :return: (rev_vocab, trunc_vocab, transformed_data
                => reverse vocabulary, truncated vocabulary, numeric sequences)
    """
    # split the text_data into word lists
    text_data = list(map(lambda x: x.split(), text_data))

    # truncate the vocabulary:
    vocab_size = len(freq_counts) if vocab_size is None else vocab_size

    trunc_vocab = dict(sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size])
    trunc_vocab = dict(enumerate(trunc_vocab.keys(), start=2))

    # add <unk> and <pad> tokens:
    trunc_vocab[1] = "<unk>"
    trunc_vocab[0] = "<pad>"

    # compute reverse trunc_vocab
    rev_trunc_vocab = dict(list(map(lambda x: (x[1], x[0]), trunc_vocab.items())))

    # transform the sentences:
    transformed_data = []  # initialize to empty list
    for sentence in text_data:
        transformed_sentence = []
        for word in sentence:
            numeric_code = rev_trunc_vocab[word] \
                if word in rev_trunc_vocab else rev_trunc_vocab["<unk>"]
            transformed_sentence.append(numeric_code)

        transformed_data.append(transformed_sentence)

    # return the truncated vocabulary and transformed sentences:
    return trunc_vocab, rev_trunc_vocab, transformed_data


def save_pickle(obj, file_name):
    """
    save the given data obj as a pickle file
    :param obj: python data object
    :param file_name: path of the output file
    :return: None (writes file to disk)
    """
    with open(file_name, 'wb') as dumper:
        pickle.dump(obj, dumper, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    """
    load a pickle object from the given pickle file
    :param file_name: path to the pickle file
    :return: obj => read pickle object
    """
    with open(file_name, "rb") as pick:
        obj = pickle.load(pick)

    return obj
