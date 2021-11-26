# NER

#### Preprocess 

1. It is processed into a list of tokens and a list of labels: X.txt.tmp 

2. use preprocess.py After processing, we get the train / dev/ test.txt: use tokenizer 

   ```bash
   python processor.py X.txt.tmp bert-base-multilingual-cased 256 > X.txt
   ```

```bash
python processor.py JA/train.conll ../bert-base-multilingual-cased 256 > JA/processed/train.txt python processor.py JA/dev.conll ../bert-base-multilingual-cased 256 > JA/processed/dev.txt python processor.py JA/test.conll ../bert-base-multilingual-cased 256 > JA/processed/test.txt
```



3. Combined with the above three TXT, run: 

   ```bash
   cat train.txt  dev.txt  test.txt  | cut -d " " -f 2 | grep -v "^$"| sort | uniq >  labels.txt  Get ner's labels.txt
   ```

## 1. Finetune

#### English EN

```bash
python run_ner.py --data_dir EN/processed/ --labels EN/processed/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir EN/output --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 400 --seed 42 --do_train --do_eval --device 1
```



#### Chinese CN

```bash
CUDA_VISIBLE_DEVICES=6 python run_ner.py --data_dir CN/peopleDaily/processed/ --labels CN/peopleDaily/processed/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir CN/peopleDaily/output --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 400 --seed 42 --do_train --do_eval  --overwrite_output_dir 
```



#### Arabic AR

```bash
CUDA_VISIBLE_DEVICES=2 python run_ner.py --data_dir AR/AQMAR/processed1/ --labels AR/AQMAR/processed1/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir AR/AQMAR/output --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 100 --seed 42 --do_train --do_eval  --overwrite_output_dir  > AR_finetune.log
```



#### HEbrew HE

```bash
CUDA_VISIBLE_DEVICES=6 python run_ner.py --data_dir HE/hebrew_ner/ --labels HE/hebrew_ner/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir HE/hebrew_ner/output_xlm --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 100 --seed 42 --do_train --do_eval  --overwrite_output_dir 
```

#### GErman DE

```bash
CUDA_VISIBLE_DEVICES=6 python run_ner.py --data_dir DE/conll2003/processed --labels DE/conll2003/processed/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir DE/conll2003/processed/output_xlm --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 100 --seed 42 --do_train --do_eval  --overwrite_output_dir 
```



#### Persian FA

```bash
CUDA_VISIBLE_DEVICES=6 python run_ner.py --data_dir FA/ArmanPersoNER/processed/ --labels FA/ArmanPersoNER/processed/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir FA/output_mbert --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 100 --seed 42 --do_train --do_eval
```



#### Japanese JA

```bash
CUDA_VISIBLE_DEVICES=6 python run_ner.py --data_dir JA/processed/ --labels JA/processed/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir JA/output --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 100 --seed 42 --do_train --do_eval > JA_finetune.log
```



#### Urdu UR

```bash
CUDA_VISIBLE_DEVICES=0 python run_ner.py --data_dir Urdu/MK-PUCIT/procssed/ --labels Urdu/MK-PUCIT/procssed/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir Urdu/output_mbert --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 100 --seed 42 --do_train --do_eval 
```



#### Dutch DU

```bah
CUDA_VISIBLE_DEVICES=6 python run_ner.py --data_dir DU/conll2002/processed/ --labels DU/conll2002/processed/labels.txt --model_name_or_path ../bert-base-multilingual-cased --output_dir DU/output_mbert --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 8 --save_steps 50 --seed 42 --do_train --do_eval 
```



## 2. Get Gradient (ABS)

#### English EN

```bash
python get_gradient.py --data_dir EN/processed/ --labels EN/processed/labels.txt --model_name_or_path ./EN/output --output_dir EN/output --max_seq_length 256 --save_steps 400 --seed 42 --do_eval
```



```bash
mv EN/output/gradient_file.pkl ./gradient_files/EN_gradient_abs.pkl
```



#### Chinese CN

```bash
python get_gradient.py --data_dir CN/peopleDaily/processed/ --labels CN/peopleDaily/processed/labels.txt --model_name_or_path CN/peopleDaily/output --output_dir CN/peopleDaily/output --max_seq_length 256 --save_steps 400 --seed 42 --do_eval
```



```bash
mv CN/peopleDaily/output/gradient_file.pkl ./gradient_files/CN_gradient_abs.pkl
```



#### Persian FA

```bash
python get_gradient.py --data_dir FA/ArmanPersoNER/processed/ --labels FA/ArmanPersoNER/processed/labels.txt --model_name_or_path FA/ArmanPersoNER/output --output_dir FA/ArmanPersoNER/output --max_seq_length 256 --save_steps 100 --seed 42 --do_eval
```



```bash
mv FA/ArmanPersoNER/output/gradient_file.pkl ./gradient_files/FA_gradient_abs.pkl
```



#### Japanese JA

```bash
python get_gradient.py --data_dir JA/processed/ --labels JA/processed/labels.txt --model_name_or_path JA/output --output_dir JA/output --max_seq_length 256  --save_steps 50 --seed 42 --do_eval
```



```bash
mv JA/output/gradient_file.pkl ./gradient_files/JA_gradient_abs.pkl
```



#### HEbrew HE

```bash
python get_gradient.py --data_dir HE/processed/ --labels HE/processed/labels.txt --model_name_or_path HE/output --output_dir HE/output --max_seq_length 256  --save_steps 50 --seed 42 --do_eval
```



```bash
mv HE/output/gradient_file.pkl ./gradient_files/HE_gradient_abs.pkl
```



#### Arabic AR

```bash
python get_gradient.py --data_dir AR/AQMAR/processed/ --labels AR/AQMAR/processed/labels.txt --model_name_or_path AR/AQMAR/output --output_dir AR/AQMAR/output --max_seq_length 256 --save_steps 100 --seed 42 --do_eval
```



```bash
mv AR/AQMAR/output/gradient_file.pkl ./gradient_files/AR_gradient_abs.pkl
```



#### Urdu UR

```bash
CUDA_VISIBLE_DEVICES=6 python get_gradient.py --data_dir Urdu/MK-PUCIT/procssed/ --labels Urdu/MK-PUCIT/procssed/label.txt --model_name_or_path Urdu/output_mbert --output_dir Urdu/output_mbert --max_seq_length 256  --save_steps 1000 --seed 42 --do_eval
```



```bash
mv UR/output/gradient_file.pkl ./gradient_files/UR_gradient_abs.pkl
```



#### Dutch DU

```bash
CUDA_VISIBLE_DEVICES=6 python get_gradient.py --data_dir DU/conll2002/processed/ --labels DU/conll2002/processed/labels.txt --model_name_or_path DU/output_mbert/ --output_dir DU/output_mbert --max_seq_length 256  --save_steps 50 --seed 42 --do_eval
```

```bash
mv DU/output/gradient_file.pkl ./gradient_files/DU_gradient_abs.pkl
```

## 3. Observe Gradient:

set languages  in observe_gradient.py and then: 

```bash
python observe_gradients.py --abs_first
```

  ***layer_normal*:**

  ```bash
python observe_gradients.py --abs_first --normal
  ```

Then get gradient matrix 12 * 12

## 4. Set Matrix Zero:

set languages and mask_num in set_matrix_zero.py and then:

```bash
python set_gradient_matrix_zero.py  --normal
```



you can also add  **--reverse** indicate mask reversely: 

```bash
python set_gradient_matrix_zero.py  --normal --reverse 
```

## 5. Lower Experiment Training (Gradient Guided Training)

Use lower_ex_generator_ori.py to generate command.

#### Same labels with 100% EN e.g.,

```bash
python lower_resources_en_ori.py --model_name_or_path ../bert-base-multilingual-cased --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 16 --seed 42 --save_steps -1 --do_train --do_eval --do_predict --source_data_dir EN/processed/ --train_data_rate 1.0 --data_dir TR/tr/processed/ --labels EN/processed/labels.txt --source_gradient_matrix_path ./gradient_files/EN_8_zero_gradient.pkl --output_dir experiment/ex_lower_resources/mask_8_EN_TR/tr_1.0/ --overwrite_output_dir --devices 2 --overwrite_cache > mask_8_EN_TR_tr.log
```



#### Different labels with 100% EN e.g.,

```bash
python lower_resources_en_ori_2labels.py --model_name_or_path ../bert-base-multilingual-cased --max_seq_length 256 --num_train_epochs 3 --per_device_train_batch_size 16 --seed 42 --save_steps -1 --do_train --do_eval --do_predict --source_data_dir EN/processed/ --train_data_rate 1.0 --data_dir FA/ArmanPersoNER/processed/ --labels FA/ArmanPersoNER/processed/labels.txt --source_gradient_matrix_path ./gradient_files/EN_8_zero_gradient.pkl --output_dir experiment/ex_lower_resources/mask_8_EN_FA/ArmanPersoNER_1.0/ --overwrite_output_dir --devices 2 --overwrite_cache > mask_8_EN_FA_ArmanPersoNER.log
```

