from pycorenlp import StanfordCoreNLP
from train import MODEL_CLASSES
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch
import numpy as np
from train_utils import get_single_features
import argparse
from sklearn.utils.extmath import softmax

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--max_seq_length", default=512)
parser.add_argument("--input_file", type=str, required=False, )
parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")


def clean_phrase(phrase):
    phrase = phrase.replace('\\n', '')
    phrase = phrase.replace("\\'s", "'s")
    phrase = phrase.lower()
    return phrase


def get_tokens(sent):
    parse = nlp.annotate(sent,
                         properties={'annotators': 'tokenize', 'outputFormat': 'json', 'ssplit.isOneSentence': True})
    tokens = [(tok['word'], tok['characterOffsetBegin'], tok['characterOffsetEnd']) for tok in parse['tokens']]
    return tokens


def get_token_indices(tokens, start_idx, end_idx):
    for i, (word, s_idx, e_idx) in enumerate(tokens):
        if s_idx <= start_idx < e_idx:
            tok_start_idx = i
        if s_idx <= end_idx <= e_idx:
            tok_end_idx = i + 1
            break

    return tok_start_idx, tok_end_idx


def evaluate_summary(article_data, summary, tokenizer, model, nlp, args):
    eval_dataset = get_single_features(summary, article_data, tokenizer, nlp, args)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)
    batch = [t for t in eval_dataloader][0]
    device = args.device
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3]
        mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
        sent_labels = batch[8]

        inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                  'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                  'num_dependency': num_dependency, 'sent_label': sent_labels, 'device': device}

        outputs = model(**inputs)
        dep_outputs = outputs[1].detach()
        dep_outputs = dep_outputs.squeeze(0)
        dep_outputs = dep_outputs[:num_dependency, :].cpu().numpy()

        input_full = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
        input_full = ' '.join(input_full).replace('[PAD]', '').strip()

        summary = input_full.split('[SEP]')[1].strip()

        print(f'Input Article:\t{input_full}')
        print(f'Generated summary:\t{summary}')

        num_negative = 0.
        for j, arc in enumerate(arcs[0]):
            arc_text = tokenizer.decode(arc)
            arc_text = arc_text.replace(tokenizer.pad_token, '').strip()

            if arc_text == '':  # for bert
                break

            softmax_probs = softmax([dep_outputs[j]])
            pred = np.argmax(softmax_probs[0])
            if pred == 0:
                num_negative += 1
            print(f'Arc:\t{arc_text}')
            print(f'Pred:\t{pred}')
            print(f'Probs:\t0={softmax_probs[0][0]}\t1={softmax_probs[0][1]}')

        print('\n')
        if num_negative > 0:
            print(f'Sent-level pred:\t0\n\n')
        else:
            print(f'Sent-level pred:\t1\n\n')


if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = args.model_dir
    model_type = args.model_type

    # set up parser
    nlp = StanfordCoreNLP('http://localhost:9000')

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    model = model_class.from_pretrained(model_dir)
    device = torch.device("cuda", args.gpu_device)
    args.device = device

    model.to(device)
    model.eval()

    input_file = open(args.input_file)
    input_data = [line.strip() for line in input_file.readlines()]

    for idx in range(0, len(input_data), 3):
        article_text = input_data[idx]
        summary = input_data[idx + 1]
        print(article_text)
        print(summary)
        evaluate_summary(article_text, summary, tokenizer, model, nlp, args)
