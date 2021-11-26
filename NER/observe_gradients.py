import os
import pickle
import argparse
import numpy as np
from scipy.stats import kendalltau, weightedtau

import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


model_languages = ["UR"] # ["EN", "CN", "AR", "JA", "TR", "TH", "FA"]


def avg_head(gradient_dict: dict, abs: bool = False):
    model_keys = list(gradient_dict.keys())
    # max_layers = int(len(model_keys) / 3)
    gradient_matrix = []
    for layer in range(0, len(model_keys), 3):  # should Q, K, V order
        K_V = gradient_dict[model_keys[layer + 1]] + gradient_dict[model_keys[layer + 2]]  # key + value
        # take average.
        head_num = len(K_V)
        # head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        if abs:
            head_vector = [np.abs(np.mean(K_V[head_index])) for head_index in range(head_num)]
        else:
            head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        gradient_matrix.append(head_vector)  # add in final

    return np.array(gradient_matrix)


def avg_head_layer_normal(gradient_dict: dict, abs: bool = False):
    model_keys = list(gradient_dict.keys())
    # max_layers = int(len(model_keys) / 3)
    gradient_matrix = []
    for layer in range(0, len(model_keys), 3):  # should Q, K, V order
        K_V = gradient_dict[model_keys[layer + 1]] + gradient_dict[model_keys[layer + 2]]  # key + value
        # take average.
        head_num = len(K_V)
        # head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        if abs:
            head_vector = [np.abs(np.mean(K_V[head_index])) for head_index in range(head_num)]
        else:
            head_vector = [np.mean(K_V[head_index]) for head_index in range(head_num)]
        # TODO: layer sum normalization
        # for each batch gradient matrix([12,12]), we normalize every head in a layer with the sum of these heads
        head_vector = [head_value / float(sum(head_vector)) for head_value in head_vector]
        gradient_matrix.append(head_vector)  # add in final

    def norm(a: np.ndarray):
        a = (a - np.min(a)) / float(np.max(a) - np.min(a))
        return a

    # TODO: use min_max normalization, map the whole gradient matrix to [0,1]
    gradient_matrix = norm(np.array(gradient_matrix))

    return gradient_matrix


def transform_to_head_average(abs_first=True, abs=True):
    root_path = "./gradient_files_xlm/" if args.xlm else "./gradient_files/"
    if abs_first:
        suffix = '_gradient_abs.pkl'
        save_suffix = '_avg.pkl'
    else:
        suffix = '_gradient.pkl'
        if abs:
            save_suffix = '_avg_abs.pkl'
        else:
            save_suffix = '_avg.pkl'
    if args.normal:
        save_suffix = save_suffix.rsplit(".", 1)[0] + "_layer_normal.pkl"

    model_file_names = [language + suffix for language in model_languages]

    # all_matrices = []
    for model_file_name in model_file_names:
        gradient_dict = pickle.load(open(root_path + model_file_name, "rb"))
        # transform the gradient dict to 12 * 12
        if args.normal:
            gradient_matrix = avg_head_layer_normal(gradient_dict, abs)
            pickle.dump(gradient_matrix,
                        open(root_path + ".".join(model_file_name.split(".")[:-1]) + save_suffix, "wb"))
        else:
            gradient_matrix = avg_head(gradient_dict, abs)
            pickle.dump(gradient_matrix,
                        open(root_path + ".".join(model_file_name.split(".")[:-1]) + save_suffix, "wb"))
    return


def load_head_average(abs_first, abs):
    root_path = "./gradient_files_xlm/" if args.xlm else "./gradient_files/"
    if abs_first:
        load_suffix = "_gradient_abs_avg.pkl"
    else:
        if abs:
            load_suffix = "_gradient_avg_abs.pkl"
        else:
            load_suffix = "_gradient_avg.pkl"

    # model_languages = ['zh', 'th', 'vi', 'en', 'de', 'es', 'fr']
    model_file_names = [language + load_suffix for language in model_languages]

    all_vectors = []
    for model_file_name in model_file_names:
        gradient_matrix = pickle.load(open(root_path + model_file_name, "rb"))
        all_vectors.append(gradient_matrix.reshape(-1))
    return model_languages, all_vectors


def mask_vector(gradient_vector, top_rate: 1):
    if top_rate >= 1:
        return gradient_vector
    mask_rate = 1 - top_rate
    mask_num = int(mask_rate * len(gradient_vector))
    mask_index = np.argsort(gradient_vector)[:mask_num]
    new_gradient_vector = gradient_vector.copy()
    new_gradient_vector[mask_index] = 0
    return new_gradient_vector


def mask_all_vector(gradient_matrix, top_rate):
    new_matrix = []
    for vector in gradient_matrix:
        new_matrix.append(mask_vector(vector, top_rate))
    return new_matrix


def get_score_matrix(language_num, matrices, score_type: str):
    if score_type == 'kendall':
        score_method = kendalltau
    elif score_type == 'weighted':
        score_method = weightedtau
    else:
        raise KeyError('Do not support such evaluation type yet ')

    score_matrix = np.zeros([language_num, language_num])
    for l1 in range(language_num):
        for l2 in range(language_num):
            score_matrix[l1, l2] = score_matrix[l2, l1] = score_method(matrices[l1], matrices[l2])[0]

    return score_matrix


def plot_heat_map(title_name, matrix, model_languages, root_path):
    fig, ax = plt.subplots(figsize=(len(model_languages), len(model_languages)))
    sns.heatmap(pd.DataFrame(np.round(matrix, 2), columns=model_languages, index=model_languages),
                annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
    ax.set_title(title_name, fontsize=12)
    ax.set_ylabel('Language', fontsize=12)
    ax.set_xlabel('Language', fontsize=12)
    # plt.show()
    plt.savefig(os.path.join(root_path, title_name + ".png"))
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--abs_first", action="store_true")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("--xlm",action="store_true")
    args = parser.parse_args()

    abs_first = args.abs_first
    abs = args.abs
    if abs_first:
        root_path = "./abs_avg_imgs"
    else:
        if abs:
            root_path = "./avg_abs_imgs"
        else:
            root_path = "./avg_imgs"
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    transform_to_head_average(abs_first, abs)
    model_languages, all_vectors = load_head_average(abs_first, abs)

    top_rates = [0.3, 0.5, 0.7, 0.9, 1]
    # top_rates = [1]
    score_types = ['kendall', 'weighted']
    for score_type in score_types:
        for top_rate in top_rates:
            gradient_matrix = mask_all_vector(all_vectors, top_rate)
            score_matrix = get_score_matrix(len(model_languages), gradient_matrix, score_type)
            plot_heat_map('Top ' + str(top_rate * 100) + '% ' + score_type, score_matrix, model_languages, root_path)
