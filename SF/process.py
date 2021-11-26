import sys
import os
import transformers

all_labels = set()
def split_words(tokens, labels):
    new_tokens, new_labels = [], []
    for token, label in zip(tokens, labels):
        new_token_seq = list(token)
        new_tokens.extend(new_token_seq)
        new_labels.append(label)
        if label.startswith('B'):
            new_labels.extend(['I' + label[1:]] * (len(new_token_seq)-1))
        else:
            new_labels.extend([label] * (len(new_token_seq)-1))
    return new_tokens, new_labels
# file = sys.argv[1]
root_path = 'SlotFilling/MultiATIS++.v0.1/data/train_dev_test'
output_root_path = 'SlotFilling/processed_data'
if not os.path.exists(output_root_path):
    os.mkdir(output_root_path)

for file_name in os.listdir(root_path):
    missed_count = 0
    in_file_path = os.path.join(root_path, file_name)
    lines = open(in_file_path).readlines()
    language = file_name.split(".tsv")[0].split("_")[1]
    if not os.path.exists(os.path.join(output_root_path, language)):
        os.mkdir(os.path.join(output_root_path, language))
    output_file_name = file_name.split(".tsv")[0].split("_")[0] + '.txt'
    output_file_path = os.path.join(os.path.join(output_root_path, language), output_file_name)
    with open(output_file_path, 'w') as outf:
        for line in lines[1:]:
            datas = line.split('\t')
            tokens = datas[1].split(" ")
            labels = datas[2].split(" ")
            try:
                assert len(tokens) == len(labels)
            except:
                missed_count += 1
                continue
            # if language in ['ZH', 'HI', 'JA']:
            #     tokens, labels = split_words(tokens, labels)
            #     assert len(tokens) == len(labels)
            for token, label in zip(tokens, labels):
                outf.write(token+' '+label+'\n')
                all_labels.add(label)
            outf.write('\n')

    print("File: ", file_name, " processed, Miss ", missed_count, " instances")

with open('./SlotFilling/processed_data/all_labels.txt', 'w') as output_f:
    for label in all_labels:
        output_f.write(label + '\n')
