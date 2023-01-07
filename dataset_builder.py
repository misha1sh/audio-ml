import torch
import random
import pymorphy2
from params import NO_PUNCT
from joblib import delayed
from utils import ProgressParallel
from navec import Navec
from razdel import tokenize, sentenize
morph = pymorphy2.MorphAnalyzer()



def empty_word_features(params):
    return torch.zeros([params["VARIANTS_CNT"], params["FEATURES_CNT"]],
                        dtype=torch.float32)

def get_word_features(word, params):
    res = empty_word_features(params)
    for i, variant in enumerate(morph.parse(word)[:params["VARIANTS_CNT"]]):
        for tag in variant.tag._grammemes_tuple:
            tag_index = params["feature_tags_dict"].get(tag, None)
            if tag_index:
                res[i][tag_index] = True
        res[i][params["VARIANT_PROB_IDX"]] = variant.score
    return res

def build_input_and_output(text, params):
    input = []
    output = []

    for i in range(params["INPUT_WORDS_CNT_LEFT"]):
        input.append(empty_word_features(params))
        output.append(NO_PUNCT)

    for token in tokenize(text):
        s = token.text
        puntuation_idx = params["PUNCTUATION_TARGET"].get(s, None)
        if puntuation_idx is not None:
            if output[-1] != NO_PUNCT:
                # we are unable to handle double punctuation yey
                continue
            output[-1] = puntuation_idx
            continue

        input.append(get_word_features(s, params))
        output.append(NO_PUNCT)

    for i in range(params["INPUT_WORDS_CNT_RIGHT"]):
        input.append(empty_word_features(params))
        output.append(NO_PUNCT)

    return torch.stack(input), torch.LongTensor(output)

def create_dataset_for_text(text, params):
    sampled_input = []
    sampled_output = []

    # input is (N_words, N_variants, N_features)
    # output is (N_words, )
    input, output = build_input_and_output(text, params)

    for i in range(params["INPUT_WORDS_CNT_LEFT"], len(input) - params["INPUT_WORDS_CNT_RIGHT"]):
        if output[i] != 0 or random.random() < 0.1:
            sampled_input.append(input[i - params["INPUT_WORDS_CNT_LEFT"]: i + params["INPUT_WORDS_CNT_RIGHT"]])
            sampled_output.append(output[i])
    if len(sampled_input) == 0: return None
    return torch.stack(sampled_input), torch.stack(sampled_output)

def create_dataset(texts, params):
    tasks = []
    for text in texts:
        tasks.append(delayed(create_dataset_for_text)(text, params))
    completed_tasks = ProgressParallel(n_jobs=16, total=len(tasks))(tasks)
    # for i, o in completed_tasks:
        # print(i.shape, o.shape)
    input, output = zip(*filter(lambda res: res is not None, completed_tasks))
    input_tensor, output_tensor = torch.cat(input), torch.cat(output)
    output_tensor = torch.nn.functional.one_hot(output_tensor, params["TARGET_CLASSES_COUNT"])
    return input_tensor, output_tensor.float()
