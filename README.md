# factuality-datasets

Contains code and dataset from the paper <a href="https://arxiv.org/pdf/2104.04302.pdf"> Annotating and Modeling Fine-grained Factuality in Summarization </a> Tanya Goyal and Greg Durrett, NAACL 2021.  

Environment base is Python 3.6. Also see requirements.txt. We used Stanford CoreNLP version 3.9.1.

# Models and Data
All models and datasets are available at: https://drive.google.com/drive/folders/18X9l3E1MWOFtLOMJNhlS8DEFDz4lsZuq?usp=sharing

## Manually Annotationed Error Types
Manually evaluated error type annotation for generated summaries from XSum and CNN/DM is included in the 'annotated_datasets' folder.

## Factuality Models and Data
The drive folder contains synthetic datasets (both generation-based and entity-based) used to train factuality models for both CNN/DM and XSum domain. These are in the 'factuality_models_datasets/training_datasets' folder. 

'factuality_models_datasets/training_datasets/XSUM-human': For XSum, additionally, the human-annotated training and test set (original data provided in <a href="https://arxiv.org/abs/2005.00661">this paper </a>) is included along with the train test splits used our paper. The corresponding tsv files contain input, summary pairs, the sentence-level factuality label as well as the arc-level factuality labels derived from the span-level annotation provided by the original paper. These files can be used directly (no further preprocessing) to train factuality models. Use train.tsv to train DAE and sentence-level models, and train_weak.tsv to train DAE-Weak model.


## Generation Models and Data
'generation_models_datasets/models' folder contains the three generation models trained on XSum that are compared in the paper, i.e. the baseline model, the loss truncation model, and the DAE-based generation model trained on subset of the tokens.

'generation_models_datasets/data': Contains the train and dev data used to train the above 3 models. The files include an additional 'output_ids' column: for each input, summary pair, this lists word indices that have been judged as non-factual by the DAE model (best ckpt model: 'factuality_models_datasets/factuality_models/DAE_xsum_human_best_ckpt.zip'). The loss corresponding to these tokens is ignored during training.


# Running Code
Download the training datasets from the google drive. The data has been preprocessed (dependency parsed with arc level factuality labels, as well as sentence level labels). You can train any of the 3 types of models by setting the model_type argument: 'electra_sentence' (sentence-level factuality model), 'electra_dae' (dependency arc entailment model) or 'electra_dae_weak' (DAE-Weak model).

### Training new factuality models
Run the following command to train models

```
python3 train.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path google/electra-base-discriminator \
    --do_train \
    --do_eval \
    --train_data_file=$DATA_DIR/train.tsv \
    --eval_data_file=$DATA_DIR/dev.tsv  \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --num_train_epochs 3.0 \
    --learning_rate 2e-5 \
    --output_dir $MODEL_DIR
```

### Running pretrained factuality models
To run the trained factuality model on the preprocessed dev files (like the 'factuality_models_datasets/training_datasets/XSUM-human/test.tsv') files, simply run the following code (you may need to change the name from test.tsv to dev.tsv). This will generate a dev_out.txt file with predicted factuality labels at both the arc-level and the sentence-level. 

```
python3 train.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path google/electra-base-discriminator \
    --input_dir $MODEL_DIR \
    --do_eval \
    --train_data_file=$DATA_DIR/train.tsv \
    --eval_data_file=$DATA_DIR/dev.tsv  \
    --per_gpu_eval_batch_size=8   \
    --output_dir $MODEL_DIR
```


To run the trained factuality model on non-preprocessed (input, summary) pairs, i.e., those which have not been dependency parsed, run the following code. This will output the predicted factuality labels at both the arc-level and the sentence-level. This code only works for model types 'electra_dae' and 'electra_dae_weak'.
```
python3 evaluate_generated_outputs.py \
        --model_type electra_dae \
        --model_dir $MODEL_DIR  \
        --input_file sample_test.txt
```
This expects an input file with the following format (see sample_test.txt for reference)
```
article1
summary1
[empty line]
article2
summary2
[empty line]
```

EDIT:
For running models on non-preprocessed data, the input file needs to be preprecessed in the following way:
1. Run both input article and summary through PTB tokenizer. 
2. Lower case both input article and summary. 

The models expect input of the above form. Not pre-processing it appropriately will hurt model performance. 
