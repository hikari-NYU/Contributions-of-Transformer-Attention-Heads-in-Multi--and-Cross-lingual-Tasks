import pickle
import numpy as np
import sys
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--mask_num", type=int, default=12)
parser.add_argument("--normal", action="store_true")  # action="store_true" type=bool,default=True
parser.add_argument("--reverse",action="store_true")  # action="store_true" type=bool,default=True
parser.add_argument("--xlm",action="store_true")
args = parser.parse_args()

lg = "UR"  # ["EN","CN","AR"]
# mask_num_lis=list(range(4,13))
# mask_num_lis = list(range(1,4))
mask_num_lis = list(range(1,13))
reverse = args.reverse  # chose whether to mask the attention head with weight from high to low

for args.mask_num in mask_num_lis:
    if args.xlm:
        file = "./gradient_files_xlm/{}_gradient_abs_avg_layer_normal.pkl" if args.normal else "./gradient_files_xlm/{}_gradient_abs_avg.pkl"
    else:
        file = "./gradient_files/{}_gradient_abs_avg_layer_normal.pkl" if args.normal else "./gradient_files/{}_gradient_abs_avg.pkl"
    file = file.format(lg)
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

    mask_num = args.mask_num

    matrix_duplicate = copy.copy(matrix)
    if reverse:
        matrix[np.argsort(matrix_duplicate)[::-1][:mask_num]] = 0  # but you should bak the matrix here
        matrix[np.argsort(matrix_duplicate)[::-1][mask_num:]] = 1
    else:
        matrix[np.argsort(matrix)[:mask_num]] = 0  # be careful! because the minimal head has been set to zero, the second line can work well as normal
        matrix[np.argsort(matrix)[mask_num:]] = 1

    wait = True

    if args.xlm:
        out = "./gradient_files_xlm/{}_".format(lg) + str(
            mask_num) + "_zero_gradient_normal.pkl" if args.normal else "./gradient_files_xlm/{}_".format(lg) + str(
            mask_num) + "_zero_gradient.pkl"
    else:
        out = "./gradient_files/{}_".format(lg) + str(
            mask_num) + "_zero_gradient_normal.pkl" if args.normal else "./gradient_files/{}_".format(lg) + str(
            mask_num) + "_zero_gradient.pkl"

    if reverse:
        out=out.rsplit(".",1)[0] + "_reverse.pkl"

    pickle.dump(matrix, open(out, "wb"))
# pickle.dump(matrix, open("./gradient_files/EN_"+str(mask_num)+"_front_zero_gradient.pkl", "wb"))
