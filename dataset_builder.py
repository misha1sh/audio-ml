import torch
import random
import pymorphy2
from params import NO_PUNCT
from joblib import delayed
from utils import ProgressParallel, download_file
from navec import Navec
import itertools
from razdel import tokenize, sentenize
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter, MosesPunctuationNormalizer
punctuation_normalizer = MosesPunctuationNormalizer('ru')

morph = pymorphy2.MorphAnalyzer()

from navec import Navec
navec_path = download_file('hudlit_12B_500K_300d_100q.tar',
        "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar")
navec = Navec.load(navec_path)

NAVEC_UNK = navec['<unk>']
NAVEC_UNK_TORCH = torch.from_numpy(NAVEC_UNK)
NAVEC_PAD_TORCH = torch.from_numpy(navec['<pad>'])

UNDEF_TOKEN = "UNDEF"
PAD_TOKEN = "PAD"


def empty_word_features(params):
    return torch.zeros([params["TOTAL_WORD_FEATURES_CNT"]],
                        dtype=torch.float32)

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
    '\'': 'PUNCT_QUOTE',
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
        word_punct = punctuation_normalizer(word).strip()
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
        res[get_navec_start_idx(params): ] = \
                torch.from_numpy(navec.get(word.lower(), NAVEC_UNK))

    return res

def build_tokens(text, return_offsets_mapping=True):
    tokens = [i for i in tokenize(text)]
    out = {}
    out["input_ids"] = [token.text for token in tokens]
    if return_offsets_mapping:
        out["offset_mapping"] = [(token.start, token.stop) for token in tokens]
    return out

def get_input_and_output_tokens(tokens, params):
    input = []
    output = []

    for i in range(params["INPUT_WORDS_CNT_LEFT"]):
        input.append(PAD_TOKEN)
        output.append(NO_PUNCT)

    next_output_id = NO_PUNCT
    for token in tokens:
        punctuation_idx = params["PUNCTUATION_TARGET"].get(token, NO_PUNCT)
        if params["RETAIN_LEFT_PUNCT"]:
            input.append(token)
            output.append(punctuation_idx)
            continue

        if punctuation_idx != NO_PUNCT:
            next_output_id = punctuation_idx
            continue

        input.append(token)
        output.append(next_output_id)
        next_output_id = NO_PUNCT

    for i in range(params["INPUT_WORDS_CNT_RIGHT"] + 1):
        input.append(PAD_TOKEN)
        output.append(NO_PUNCT)

    return input, torch.LongTensor(output)


def calculate_word_features_for_tokens(input, params):
    input = [get_word_features(s, params) for s in input]
    return torch.stack(input)

# randomly replaces tokens with UNDEF. Required for working with shapely
def infect_tokens(features, tokens_list, params):
    assert len(features) == len(tokens_list)

    # for shapely we want to infect tokens in linear distribution
    infected_tokens_count = random.randrange(len(tokens_list))
    normal_tokens_count = len(tokens_list) - infected_tokens_count
    infected_tokens_bits = [True for i in range(infected_tokens_count)] + \
                           [False for i in range(normal_tokens_count)]
    random.shuffle(infected_tokens_bits)

    features_new = torch.clone(features)
    tokens_list_new = tokens_list.copy()
    for i, is_infected in enumerate(infected_tokens_bits):
        if is_infected:
            features_new[i] = undef_word_features(params)
            tokens_list_new[i] = UNDEF_TOKEN
    return features_new, tokens_list_new


def create_dataset_for_text(text, params):
    sampled_input = []
    sampled_output = []
    is_infected = []
    texts = []

    # input is (N_words, N_variants, N_features)
    # output is (N_words, )
    if params['type'] == 'nerus':
        tokens = [token.text for token in text.tokens]
    else:
        tokens = build_tokens(text, False)['input_ids']

    input_tokens, output = get_input_and_output_tokens(tokens, params)
    # print(list([ (j, i.item()) for i, j in zip(output, input_tokens) ]) )
    input = calculate_word_features_for_tokens(input_tokens, params)
    WORDS_LEFT = params["INPUT_WORDS_CNT_LEFT"]
    WORDS_RIGHT = params["INPUT_WORDS_CNT_RIGHT"]
    for i in range(WORDS_LEFT, len(input) - WORDS_RIGHT):
        take_sample = output[i] != 0 or random.random() < 0.1
        if not take_sample: continue

        if params["RETAIN_LEFT_PUNCT"]:
            inp = torch.zeros_like(input[i - WORDS_LEFT: i + WORDS_RIGHT])
            inp[:WORDS_LEFT] =  input[i - WORDS_LEFT: i]
            inp_tokens = input_tokens[i - WORDS_LEFT: i]
            cnt_tokens = WORDS_LEFT
            for j in range(i, len(input)):
                assert cnt_tokens == len(inp_tokens)
                if cnt_tokens == WORDS_LEFT + WORDS_RIGHT:
                    break
                if output[j] == 0:
                    inp[cnt_tokens] = input[j]
                    inp_tokens.append(input_tokens[j])
                    cnt_tokens += 1
            assert cnt_tokens == len(inp_tokens)
            if len(inp_tokens) != WORDS_LEFT + WORDS_RIGHT:
                print(j, len(input), len(inp_tokens), WORDS_LEFT + WORDS_RIGHT)
                print(input_tokens[i - WORDS_LEFT: i + WORDS_RIGHT])
                assert len(inp_tokens) == WORDS_LEFT + WORDS_RIGHT

        else:
            inp_tokens = input_tokens[i - WORDS_LEFT: i + WORDS_RIGHT]
            inp = input[i - WORDS_LEFT: i + WORDS_RIGHT]

        out = output[i]
        infect = random.random() < params['INFECTED_TEXT_PROB']
        if infect:
            inp, inp_tokens = infect_tokens(inp, inp_tokens, params)
        # text = " ".join(inp_tokens[:WORDS_LEFT]) + " #" + str(params["ID_TO_PUNCTUATION"][out.item()]) + "# " + \
        #        " ".join(inp_tokens[WORDS_LEFT: WORDS_LEFT + WORDS_RIGHT])

        text = inp_tokens[:WORDS_LEFT] + [" #" + str(params["ID_TO_PUNCTUATION"][out.item()]) + "# "] + \
               inp_tokens[WORDS_LEFT: WORDS_LEFT + WORDS_RIGHT]


        is_infected.append(infect)
        texts.append(text)
        sampled_input.append(inp)
        sampled_output.append(out)


    if len(sampled_input) == 0: return None
    return torch.stack(sampled_input), torch.stack(sampled_output), texts, torch.BoolTensor(is_infected)

def create_dataset(texts, params):
    tasks = []
    for text in texts:
        tasks.append(delayed(create_dataset_for_text)(text, params))
    if len(tasks) > 10:
        completed_tasks = ProgressParallel(n_jobs=16, total=len(tasks))(tasks)
    else:
        completed_tasks = [task[0](*task[1], **task[2]) for task in tasks]
    # for i, o in completed_tasks:
        # print(i.shape, o.shape)
    input, output, texts_res, is_infected = zip(*filter(lambda res: res is not None, completed_tasks))
    input_tensor, output_tensor = torch.cat(input), torch.cat(output)
    is_infected = torch.cat(is_infected)
    texts_res = [item for sublist in texts_res for item in sublist]
    output_tensor = torch.nn.functional.one_hot(output_tensor, params["TARGET_CLASSES_COUNT"])
    return input_tensor, output_tensor.float(), texts_res, is_infected
