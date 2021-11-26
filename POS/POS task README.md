
# POS

- Use Universal Dependencies Dataset.

1. replace .conllu file to .txt
```sh
rm -rf *.txt
mv *dev.conllu dev.txt
mv *train.conllu train.txt
mv *test.conllu test.txt
```



2. Process merge labels: (HE, AR, TR, FA, DE)

```sh
python process_merge_word.py YOUR_FILE_PLACE/[train|dev|test].txt
```

TODO DE



3. Process AR labels # delete stubborn labels

python process_ar_labels.py AR_FILE_PLACE/[train|dev|test].txt



## 1. Finetune

#### AR # UD_Arabic-NYUAD
python run_pos.py --task_type POS --data_dir ./AR/UD_Arabic/UD_Arabic-PADT --model_name_or_path ../bert-base-multilingual-cased/ --output_dir ./experiment/finetune/ar_padt --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### CN # UD_Chinese-GSDSimp
python run_pos.py --task_type POS --data_dir ./CN/UD_Chinese/UD_Chinese-GSDSimp --model_name_or_path ../bert-base-multilingual-cased/ --output_dir ./experiment/finetune/cn_gsdsimp --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### EN # UD_English-EWT
python run_pos.py --task_type POS --data_dir ./EN/UD_English/UD_English-EWT --model_name_or_path ../bert-base-multilingual-cased/ --output_dir ./experiment/finetune/en_ewt --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### JA # UD_Japanese-GSD
python run_pos.py --task_type POS --data_dir ./JA/UD_Japanese/UD_Japanese-GSD --model_name_or_path ../bert-base-multilingual-cased/ --output_dir ./experiment/finetune/ja_gsd --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### FA # UD_Persian-Seraji
python run_pos.py --task_type POS --data_dir ./FA/UD_Persian/UD_Persian-Seraji  --model_name_or_path ../bert-base-multilingual-cased/ --output_dir ./experiment/finetune/fa_seraji --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### TR # UD_Turkish-IMST
python run_pos.py --task_type POS --data_dir ./TR/UD_Turkish/UD_Turkish-IMST  --model_name_or_path ../bert-base-multilingual-cased/ --output_dir ./experiment/finetune/tr_imst --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### RU # UD_Russian-GSD
python run_pos.py --task_type POS --data_dir ./RU/UD_Russian/UD_Russian-GSD  --model_name_or_path ../bert-base-multilingual-cased/ --output_dir ./experiment/finetune/ru_gsd --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### HE # UD_Hebrew-HTB

nohup python run_pos.py --task_type POS --data_dir ./HE/UD_Hebrew/UD_Hebrew-HTB/ --model_name_or_path ../bert-base-multilingual-cased/ --output_dir /data_local/yutianyu/zhangkai/mbert/POS/experiment/finetune/he_htb --max_seq_length  256 --num_train_epochs 3 --per_gpu_train_batch_size 16 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 0 > finetune_he.log &



## 2. Get Gradient (ABS)

#### AR
python get_gradient.py --task_type POS --data_dir ./AR/UD_Arabic/UD_Arabic-PADT --model_name_or_path ./experiment/finetune/ar_padt --output_dir ./gradient_files/ --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/AR_gradient_abs.pkl
#### CN
python get_gradient.py --task_type POS --data_dir ./CN/UD_Chinese/UD_Chinese-GSDSimp --model_name_or_path ./experiment/finetune/cn_gsdsimp --output_dir ./gradient_files/ --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/CN_gradient_abs.pkl
#### EN
python get_gradient.py --task_type POS --data_dir ./EN/UD_English/UD_English-EWT --model_name_or_path ./experiment/finetune/en_ewt --output_dir ./gradient_files/ --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/EN_gradient_abs.pkl
#### JA
python get_gradient.py --task_type POS --data_dir ./JA/UD_Japanese/UD_Japanese-GSD --model_name_or_path ./experiment/finetune/ja_gsd --output_dir ./gradient_files/ --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/JA_gradient_abs.pkl
#### FA
python get_gradient.py --task_type POS --data_dir ./FA/UD_Persian/UD_Persian-Seraji --model_name_or_path ./experiment/finetune/fa_seraji --output_dir ./gradient_files/ --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/FA_gradient_abs.pkl
#### TR
python get_gradient.py --task_type POS --data_dir ./TR/UD_Turkish/UD_Turkish-IMST --model_name_or_path ./experiment/finetune/tr_imst --output_dir ./gradient_files/ --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/TR_gradient_abs.pkl
#### RU
python get_gradient.py --task_type POS --data_dir ./RU/UD_Russian/UD_Russian-GSD --model_name_or_path ./experiment/finetune/ar_padt --output_dir ./gradient_files/ru_gsd --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/RU_gradient_abs.pkl

#### HE

python get_gradient.py --task_type POS --data_dir ./HE/UD_Hebrew/UD_Hebrew-HTB/ --model_name_or_path /data_local/yutianyu/zhangkai/mbert/POS/experiment/finetune/ar_padt --output_dir ./gradient_files/he_htb --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/HE_gradient_abs.pkl

## 3. Observe Gradient:
1. set languages in observe_gradient.py
python observe_gradient.py --abs_first

Then get gradient matrix 12 * 12

## 4. Set Matrix Zero:
1. python set_gradient_matrix_zero.py [number_of_zero_gradient] # sorted from the least.

## 5. Lower Experiment Training (Gradient Guided Training)

Use lower_ex_generator_ori.py to generate command.

#### Same labels with 100% EN e.g.,

python lower_resources_en_ori.py --task_type POS --model_name_or_path ../bert-base-multilingual-cased --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 16 --seed 42 --save_steps -1 --do_train --do_eval --do_predict --overwrite_cache --source_data_dir EN/UD_English/UD_English-EWT --train_data_rate 0.0 --data_dir RU/UD_Russian/UD_Russian-GSD --source_gradient_matrix_path ./gradient_files/EN_0_zero_gradient.pkl --output_dir experiment/ex_lower_resources/mask_0_EN_RU/gsd_0.0/ --overwrite_output_dir --devices 4
























