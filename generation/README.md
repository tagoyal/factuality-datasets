# Generation Models training and evaluation

This folder contains code to train and generate from a DAE-based generation model trained on a subset of tokens. Corresponds to Section 6.2 of the paper.
The steps for training a generation model with partial loss is the following:

### Step1: Given an article, summary, identifying non-factual spans using DAE.

1) Run preprocessing.py to create data of the form required by DAE models. The file expects an input tsv with two columns 'article' and 'summary'. See sample file for reference. The output file should be named **dev.tsv**

```
python preprocessing.py  --input_file $INPUT_FILE  --output_file $OUTPUT_FILE
```

2) Run DAE model, using the train.py (but without setting the --do_train flag). Here, the input_dir points to the location of the best checkpoint model. Note that the evaluation is always run on a file named **dev.tsv**. That is, the DAE model is run on the file '--eval_data_file=$DATA_DIR/dev.tsv'. Please name the files accordingly. 

```
python3 train.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path google/electra-base-discriminator \
    --input_dir $MODEL_DIR \
    --do_eval \
    --eval_data_file=$DATA_DIR/dev.tsv  \
    --per_gpu_eval_batch_size=8   \
    --output_dir $OUTPUT_DIR
```

This generates a file 'dev_out.txt' used in the next step . 

### Step2: Creating generation training data w/ information about the non-factual tokens

Run create_data_for_generation.py. Modify the code to point to correct paths for dev_out.txt and the output_folder.

```
python create_data_for_generation.py
```

This generates tsv files. Run this on the dev_out.txt files corresponding to BOTH the train and dev sets. 

### Step3: Training models

The code in train_generation.py can be used to train the DAE-based generation model.
```
python3 train_generation.py \
    --model_type bart_partial \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --train_data_file=$DATA_DIR/train.tsv \
    --eval_data_file=$DATA_DIR/dev.tsv \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=4   \
    --gradient_accumulation_steps=2 \
    --num_train_epochs 5.0 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --partial_loss
```
Remove flag --partial_loss to train the baseline model **WITHOUT** token subsampling.  


### Generating from trained models 

Set input_dir and the generate flag.
```
python3 train_generation.py \
    --model_type bart_partial \
    --model_name_or_path facebook/bart-large \
    --input_dir $MODEL_DIR \
    --eval_data_file=$DATA_DIR/dev.tsv \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=4   \
    --gradient_accumulation_steps=2 \
    --num_train_epochs 5.0 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_DIR \
    --generate
```

