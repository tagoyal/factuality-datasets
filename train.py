import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import train_utils
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange
from transformers import glue_compute_metrics as compute_metrics
from sklearn.utils.extmath import softmax
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import  precision_score, recall_score

from transformers import (
    AdamW,
    ElectraConfig,
    ElectraTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "electra_sentence": (ElectraConfig, train_utils.ElectraBasicModel, ElectraTokenizer),
    "electra_dae": (ElectraConfig, train_utils.ElectraDependencyModel, ElectraTokenizer),
    "electra_dae_weak": (ElectraConfig, train_utils.ElectraConstModelTwoClass, ElectraTokenizer),
    }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_checkpoints(args, output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = train_utils.load_and_cache_examples(args, tokenizer, evaluate)
    return dataset


def compute_metrics_balanced(preds, golds):
    n_0 = 0.
    d_0 = 0.
    n_1 = 0.
    d_1 = 0.
    for p, g in zip(preds, golds):
        if g == 0:
            if p == 0:
                n_0 += 1
            d_0 += 1
        elif g == 1:
            if p == 1:
                n_1 += 1
            d_1 += 1

    acc_0 = n_0 / d_0
    acc_1 = n_1 / d_1

    return {'acc': (acc_0 + acc_1) / 2}


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids_sent = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3]
            mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
            sent_labels = batch[8]

            inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                      'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                      'num_dependency': num_dependency, 'sent_label': sent_labels, 'device': args.device}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids_sent = sent_labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids_sent = np.append(out_label_ids_sent, sent_labels.detach().cpu().numpy(), axis=0)

    f_out = open(os.path.join(eval_output_dir, 'dev_out.txt'), 'w')
    k = 0
    sent_pred = []
    dep_pred = []
    dep_gold = []
    nb_eval_steps = 0
    for batch in eval_dataloader:
        nb_eval_steps += 1
        for inp, p_mask, arc_list, head_ids, child_ids in zip(batch[0], batch[4], batch[7], batch[3], batch[2]):
            # text = tokenizer.decode(inp)
            tokens = tokenizer.convert_ids_to_tokens(inp)
            article_len = tokens.index('[SEP]') + 1
            text_article = tokens[1:article_len - 1]  # removing [CLS] and [SEP]

            summary = tokens[article_len:]  # has all the pad tokens also
            if '[PAD]' in summary:
                summary_len = summary.index('[PAD]')
                summary = summary[:summary_len - 1]
            else:
                summary = summary[:-1]

            text_article_cleaned = ' '.join(text_article).replace(' ##', '')
            summary_cleaned = ' '.join(summary).replace(' ##', '')

            f_out.write(text_article_cleaned + '\n')
            f_out.write(summary_cleaned + '\n')
            num_negative = 0

            if args.model_type == 'electra_sentence':
                sent_pred_curr_prob = softmax([preds[k]])
                sent_pred_curr = np.argmax(sent_pred_curr_prob)
                sent_pred.append(sent_pred_curr)
                f_out.write('sent gold:\t%s\n' % str(out_label_ids_sent[k]))
                f_out.write('sent pred:\t%s\n\n' % str(sent_pred_curr))

            elif 'electra_dae' in args.model_type:
                for j, arc in enumerate(arc_list):
                    arc_text = tokenizer.decode(arc)
                    arc_text = arc_text.replace(tokenizer.pad_token, '').strip()
                    mask = int(p_mask[j])

                    if arc_text == '':  # for bert
                        break

                    pred_temp = softmax([preds[k][j]])

                    if mask == 1:
                        gold = 1
                    else:
                        gold = 0

                    pred = np.argmax(pred_temp)

                    dep_pred.append(pred)
                    dep_gold.append(gold)

                    if pred == 0:
                        num_negative += 1

                    f_out.write(arc_text + '\n')
                    f_out.write('gold:\t' + str(gold) + '\n')
                    f_out.write('pred:\t' + str(pred) + '\n')
                    f_out.write(str(pred_temp[0][0]) + '\t' + str(pred_temp[0][1]) + '\n\n')

                f_out.write('sent gold:\t' + str(out_label_ids_sent[k]) + '\n')
                if num_negative > 0:
                    f_out.write('sent_pred:\t0\n\n')
                    sent_pred.append(0)
                else:
                    f_out.write('sent_pred:\t1\n\n')
                    sent_pred.append(1)

            k += 1

    f_out.close()

    if args.model_type in ['electra_dae', 'electra_dae_weak']:
        dep_pred = np.array(dep_pred)
        dep_gold = np.array(dep_gold)
        sent_pred = np.array(sent_pred)
        prec = precision_score(dep_pred, dep_gold, pos_label=0)
        recall = recall_score(dep_pred, dep_gold, pos_label=0)
        f1 = f1_score(dep_pred, dep_gold, pos_label=0)
        print(prec)
        print(recall)
        print(f1)

        result_dep = compute_metrics('qqp', dep_pred, dep_gold)
        balanced_acc = balanced_accuracy_score(y_true=out_label_ids_sent, y_pred=sent_pred)
        result = {'acc': balanced_acc}
    else:
        result_dep = {}
        balanced_acc = balanced_accuracy_score(y_true=out_label_ids_sent, y_pred=sent_pred)
        result = {'acc': balanced_acc}

    print(result_dep)
    print(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result_dep.keys()):
            logger.info("dep level %s = %s", key, str(result_dep[key]))
            writer.write("dep level  %s = %s\n" % (key, str(result_dep[key])))
        for key in sorted(result.keys()):
            logger.info("sent level %s = %s", key, str(result[key]))
            writer.write("sent level  %s = %s\n" % (key, str(result[key])))
        writer.write('\n')

    if args.model_type == 'electra_dep':
        return result_dep
    else:
        return result


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size

    num_neg = 0.
    num_pos = 0.
    for tensor in train_dataset:
        sent_label = int(tensor[9])
        if sent_label == 0:
            num_neg += 1
        else:
            num_pos += 1
        #print(sent_label)

    weights = []
    w_neg = (num_pos * 10) / (num_pos + num_neg)
    w_pos = (num_neg * 10) / (num_pos + num_neg)
    for tensor in train_dataset:
        sent_label = int(tensor[9])
        if sent_label == 0:
            weights.append(w_neg)
        else:
            weights.append(w_pos)

    #train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights) * 5)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, tr_loss_sent, logging_loss, logging_loss_sent = 0.0, 0.0, 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    acc_prev = 0.

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(args.device) for t in batch)
            input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3],
            mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
            sent_labels = batch[8]

            inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                      'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                      'num_dependency': num_dependency, 'sent_label': sent_labels, 'device': args.device}

            model.train()
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    logs = {}
                    loss_scalar_dep = (tr_loss - logging_loss) / args.save_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar_dep
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step, 'epoch': epoch_iterator.n}}))
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # Evaluation
                    result = evaluate(args, model, tokenizer, eval_dataset)

                    save_checkpoints(args, args.output_dir, model, tokenizer)

                    if result['acc'] > acc_prev:
                        acc_prev = result['acc']
                        # Save model checkpoint best
                        output_dir = os.path.join(args.output_dir, "model-best")
                        save_checkpoints(args, output_dir, model, tokenizer)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

    evaluate(args, model, tokenizer, eval_dataset)
    save_checkpoints(args, args.output_dir, model, tokenizer)

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        required=True,
        help="Evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size evaluation.", )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--include_sentence_level", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--seed", type=int, default=100, help="random seed for initialization")

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.n_gpu = 1  # no multi gpu support right now.
    device = torch.device("cuda", args.gpu_device)
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, 'model.log')
    )

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.input_dir is not None:
        print('loading model')
        tokenizer = tokenizer_class.from_pretrained(args.input_dir)
        model = model_class.from_pretrained(args.input_dir)
    else:
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)

    model.to(args.device)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    evaluate(args, model, tokenizer, eval_dataset)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
