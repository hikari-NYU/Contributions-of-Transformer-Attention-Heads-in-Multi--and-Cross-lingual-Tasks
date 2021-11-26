import pickle
import numpy as np
import sys

language = sys.argv[1]

file = "./gradient_files/" + language.upper() + "_gradient_abs_avg_layer_norm.pkl"
# file = "./XNLI/gradient_files/EN_12_zero_gradient.pkl"
# file = "./gradient_files/EN_norm_gradient_matrix.pkl"
# file = "./gradient_files/EN_norm_1000_gradient_matrix.pkl"

matrix = pickle.load(open(file, "rb"))

# todo set less 12 to 0 and others to 1
head_mask = matrix
head_mask = head_mask.reshape(144, 1)
# head_mask = (head_mask - np.min(head_mask)) / (np.max(head_mask) - np.min(head_mask))
wait = True

matrix = matrix.reshape(144)

# matrix[-24:] = 1

mask_num = int(sys.argv[2])

matrix[np.argsort(matrix)[:mask_num]] = 0
matrix[np.argsort(matrix)[mask_num:]] = 1

pickle.dump(matrix, open("./gradient_files/" + language.upper() + "_"+str(mask_num)+"_zero_gradient_layer_norm.pkl", "wb"))
# pickle.dump(matrix, open("./gradient_files/EN_"+str(mask_num)+"_front_zero_gradient.pkl", "wb"))
