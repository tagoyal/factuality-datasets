#!/bin/sh
cd stanford-corenlp-full-2018-02-27
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 &
cd ../factuality-datasets
../bin/python3.6 evaluate_generated_outputs.py --model_type electra_dae --model_dir ../DAE_xsum_human_best_ckpt --input_file sample_test.txt