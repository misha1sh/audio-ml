from utils import reverse_dict
import string


NO_PUNCT = 0


def build_params(params):
    params["feature_tags_dict"] = {tag: i for i, tag in enumerate(sorted([
            item for sublist in params["feature_tags_array"] for item in sublist]))}
    params.pop("feature_tags_array")

    params["ID_TO_PUNCTUATION"] = reverse_dict(params["PUNCTUATION_TARGET"],
                                            priority_for_duplicates=['.'])
    params["VARIANT_FEATURES_CNT"] = sum([
                len(params["feature_tags_dict"]), # noun, case, number, ...
                1 # variant score
        ])
    params["EMBEDDING_FEATURES_CNT"] = sum([
                300 if params["USE_NAVEC"] else 0
    ])
    params["TOTAL_WORD_FEATURES_CNT"] = params["VARIANTS_CNT"] * params["VARIANT_FEATURES_CNT"] + \
                        params['EMBEDDING_FEATURES_CNT']


    params["VARIANT_PROB_IDX"] = len(params["feature_tags_dict"])

    params["PUNCTUATION_ALL"] = string.punctuation

    params["INPUT_WORDS_CNT_RIGHT"] = params["INPUT_WORDS_CNT"] // 2
    params["INPUT_WORDS_CNT_LEFT"] = params["INPUT_WORDS_CNT"] - params["INPUT_WORDS_CNT_RIGHT"]

    return params
