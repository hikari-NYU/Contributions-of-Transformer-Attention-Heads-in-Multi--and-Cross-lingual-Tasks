# Slot Filling

#### Preprocess 

1. process to token classification type

   python process.py

2. run：
   cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
   to get labels.txt of  Slot Filling

## 1. Finetune

#### English EN

CUDA_VISIBLE_DEVICES=0 python run_sf.py --data_dir processed_data/EN/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/EN --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 0

#### Chinese ZH

CUDA_VISIBLE_DEVICES=0 python run_sf.py --data_dir processed_data/ZH/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/ZH --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 0



#### DE

CUDA_VISIBLE_DEVICES=1 python run_sf.py --data_dir processed_data/DE/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/DE --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 1

#### ES

CUDA_VISIBLE_DEVICES=2 python run_sf.py --data_dir processed_data/ES/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/ES --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 2

#### FR

CUDA_VISIBLE_DEVICES=3 python run_sf.py --data_dir processed_data/FR/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/FR --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 3

#### HI

CUDA_VISIBLE_DEVICES=4 python run_sf.py --data_dir processed_data/HI/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/HI --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 4

#### Japanese JA

CUDA_VISIBLE_DEVICES=5 python run_sf.py --data_dir processed_data/JA/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/JA --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 5

#### PT

CUDA_VISIBLE_DEVICES=6 python run_sf.py --data_dir processed_data/PT/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/PT --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 6

#### Turkish TR

CUDA_VISIBLE_DEVICES=7 python run_sf.py --data_dir processed_data/TR/ --labels processed_data/all_labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir experiment/finetune/TR --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps -1 --seed 42 --do_train --do_eval --do_predict --devices 7



## 2. Get Gradient (ABS)

#### English EN

python get_gradient.py --task_type NER --labels processed_data/all_labels.txt --data_dir processed_data/EN/ --model_name_or_path ./experiment/finetune/EN --output_dir ./gradient_files/ --max_seq_length 256 --seed 42 --do_predict --gradient_save_path ./gradient_files/EN_gradient_abs.pkl

## 3. Observe Gradient:

1. set languages in observe_gradient.py
   python observe_gradient_layer_norm.py --abs_first

Then get gradient matrix 12 * 12

## 4. Set Matrix Zero:

1. python set_gradient_matrix_zero_layer_norm.py [number_of_zero_gradient] # sorted from the least.


## 5. Lower Experiment Training (Gradient Guided Training)

Use lower_ex_generator_ori.py to generate command.

#### Different labels with 100% EN e.g.,

CUDA_VISIBLE_DEVICES=1 python lower_resources_en_ori.py --task_type NER --model_name_or_path ../bert-base-multilingual-cased --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 16 --seed 42 --save_steps -1 --do_train --do_eval --do_predict --overwrite_cache --source_data_dir processed_data/EN --train_data_rate 1.0 --labels  processed_data/all_labels.txt --data_dir processed_data/DE --source_gradient_matrix_path ./gradient_files/EN_1_zero_gradient_layer_norm.pkl --output_dir /data_local/yutianyu/zhangkai/mbert/POS/experiment/ex_lower_resources/layer_norm_mask_1_EN_DE/1.0/ --overwrite_output_dir --devices 1 & 




