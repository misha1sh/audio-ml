import micropip

import jsinfer

micropip.add_mock_package("docopt", "0.6.2", modules = {
    "docopt": """
        docopt = 1
    """
})
await micropip.install("pymorphy3")
await micropip.install("numpy")
# await micropip.install("sacremoses")
await micropip.install("navec")
await micropip.install("setuptools")

import numpy as np
import random
import pymorphy3
import numpy as np
import math

NO_PUNCT = 0
from navec import Navec
import itertools
# from sacremoses import MosesPunctNormalizer
from pyodide.http import pyfetch
import os

# punctuation_normalizer = MosesPunctNormalizer('ru')

morph = pymorphy3.MorphAnalyzer()

async def download_file(file, url):
    file_path = os.path.join("./", file)
    if os.path.isfile(file_path):
        return file_path
    print("donwloading", file, "to", file_path)
    # url = BASE_URL + file
    response = await pyfetch(url)
    with open(file, "wb") as f:
        f.write(await response.bytes())
    return file_path

navec_path = await download_file('hudlit_12B_500K_300d_100q.tar',
        "http://localhost:3000/hudlit_12B_500K_300d_100q.tar")
navec = Navec.load(navec_path)

params_path = await download_file('params.dill',
                        "http://localhost:3000/params.dill")

NUMPY_DTYPE = float
NAVEC_UNK = navec['<unk>']
NAVEC_UNK_TORCH = NAVEC_UNK
NAVEC_PAD_TORCH = navec['<pad>']

UNDEF_TOKEN = "UNDEF"
PAD_TOKEN = "PAD"


def empty_word_features(params):
    return np.zeros([params["TOTAL_WORD_FEATURES_CNT"]],
                        dtype=NUMPY_DTYPE)

def get_navec_start_idx(params):
    return params['VARIANT_FEATURES_CNT'] * params['VARIANTS_CNT']

def pad_word_features(params):
    res = empty_word_features(params)
    res[get_navec_start_idx(params): ] = NAVEC_PAD_TORCH
    return res

def undef_word_features(params):
    res = empty_word_features(params)
    res[get_navec_start_idx(params): ] = NAVEC_UNK_TORCH
    return res


PNCT_TAGS = {
    '.': 'PUNCT_DOT',
    '!': 'PUNCT_DOT',
    '?': 'PUNCT_DOT',
    ',': 'PUNCT_COMMA',
    '-': 'PUNCT_DASH',
    '.':'PUNCT_DOT',
    '"': 'PUNCT_QUOTE',
    #'\\'': 'PUNCT_QUOTE',
    '(': 'PUNCT_LEFT_PARENTHESIS',
    ')': 'PUNCT_RIGHT_PARENTHESIS',
}

def get_word_features(word, params):
    if word == PAD_TOKEN:
        return pad_word_features(params)
    if word == UNDEF_TOKEN:
        return undef_word_features(params)

    additional_tags = []

    res = empty_word_features(params)
    if not str.isalpha(word[0]):
        # word_punct = punctuation_normalizer(word).strip()
        word_punct = word.strip()[0]
        if word_punct in PNCT_TAGS:
            additional_tags.append(PNCT_TAGS[word_punct])

    if str.isupper(word[0]):
        additional_tags.append('CAPITALIZED')

    use_navec = True

    variant_features_cnt = params['VARIANT_FEATURES_CNT']
    for i, variant in enumerate(morph.parse(word)[:params["VARIANTS_CNT"]]):
        tags = variant.tag._grammemes_tuple

        for tag in itertools.chain(tags, additional_tags):
            tag_index = params["feature_tags_dict"].get(tag, None)
            if tag_index:
                res[i * variant_features_cnt + tag_index] = True
            if i == 0 and tag in params['CUT_NAVEC_TAGS_SET']:
                use_navec = False
        res[i * variant_features_cnt + params["VARIANT_PROB_IDX"]] = variant.score


    if params['USE_NAVEC'] and use_navec:
        res[get_navec_start_idx(params): ] = navec.get(word.lower(), NAVEC_UNK)

    return res

def calculate_word_features_for_tokens(input, params):
    input = [get_word_features(s, params) for s in input]
    return np.stack(input)


def onnx_model_runner(path):
    ort_sess = ort.InferenceSession(path)
    def func(input):
        return ort_sess.run(None, {'input': np.array(input) })[0]
    return func

async def infer(text):
    assert params["RETAIN_LEFT_PUNCT"] #

    unpadded_tokens = text.split(' ')
    unpadded_tokens = list(filter(lambda x: len(x) > 0, unpadded_tokens))
    tokens = [PAD_TOKEN] * params['INPUT_WORDS_CNT_LEFT'] + unpadded_tokens + [PAD_TOKEN] * (params["INPUT_WORDS_CNT_RIGHT"] + 1)
    features = calculate_word_features_for_tokens(tokens, params)

    res = ""

    i = params['INPUT_WORDS_CNT_LEFT']
    while i < len(tokens) - params['INPUT_WORDS_CNT_RIGHT']:
        tokens_for_batch = tokens[i - params['INPUT_WORDS_CNT_LEFT']: i + params['INPUT_WORDS_CNT_RIGHT']]

        tokens_for_batch_copy = tokens_for_batch.copy()
        tokens_for_batch_copy.insert(params['INPUT_WORDS_CNT_LEFT'], '?')
        # print(" ".join(tokens_for_batch_copy))


        features_for_batch = features[i - params['INPUT_WORDS_CNT_LEFT']: i + params['INPUT_WORDS_CNT_RIGHT']]
        features_for_batch = np.stack((features_for_batch, ))
        output_probs = await jsinfer.infer(features_for_batch)
        break
        punct_idx = np.argmax(output_probs).item()
        punct = params["ID_TO_PUNCTUATION"][punct_idx]

        # print(punct)

        # punct = '.'

        if punct != '$empty':
            res += punct
            if tokens[i] != 'PAD':
                res += " " + tokens[i]
            tokens.insert(i, punct)
            features = torch.cat((features[:i],
                    torch.stack((get_word_features(punct, params), )),
                    features[i:]), 0)
            i += 2
        else:
            if tokens[i] != 'PAD':
                res += " " + tokens[i]
            i += 1

    return res.strip()

