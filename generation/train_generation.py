import argparse
import json
import logging
import os
import csv
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import train_generation_utils


from transformers import (
    AdamW,
    BartConfig,
    BartTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


logger = logging.getLogger(__name__)

MODEL_CLASSES = {"bart_partial": (BartConfig, train_generation_utils.BartForConditionalGenerationCustom,
                                  BartTokenizer),}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_checkpoints(args, output_dir, model, tokenizer, optimizer, scheduler):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict:
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss_sentence = 0.0
    nb_eval_steps = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            input_ids, attention, decoder_ids = batch[0], batch[1], batch[2]
            decoder_attention, decoder_output_ids = batch[3], batch[4]

            inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                      'decoder_attention_mask': decoder_attention, 'lm_labels': decoder_output_ids, 'generate': False}

            outputs = model(**inputs)
            tmp_eval_loss_sentence = outputs[0]
            eval_loss_sentence += tmp_eval_loss_sentence.item()
            nb_eval_steps += 1

    if args.generate:
        f_out = open(os.path.join(eval_output_dir, 'dev_out.txt'), 'w')
        print(eval_output_dir)
        k = 0

        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_attention_mask, decoder_ids = batch[0], batch[1], batch[2]

            for j in range(input_ids.shape[0]):

                gold = tokenizer.decode(decoder_ids[j], skip_special_tokens=True)
                input = tokenizer.decode(input_ids[j], skip_special_tokens=True)

                gen = model.generate(input_ids[j].unsqueeze(0), attention_mask=input_attention_mask[j].unsqueeze(0),
                                     num_beam=6, length_penalty=2, no_repeat_ngram_size=3, max_length=200, min_length=10,
                                     decoder_start_token_id=tokenizer.bos_token_id)
                gen = tokenizer.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                print(gen.strip())
                if len(gen) == 0:
                    continue

                f_out.write(input + '\n')
                f_out.write(gold + '\n')
                f_out.write(gen.strip() + '\n\n')

            k += 1

        f_out.close()

    result = {'loss': eval_loss_sentence/nb_eval_steps}
    print(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write('\n')
    return result


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

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
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, tr_loss_sent, logging_loss, logging_loss_sent = 0.0, 0.0, 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    torch.cuda.empty_cache()
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = tuple(t.to(args.device) for t in batch)

            input_ids, attention, decoder_ids = batch[0], batch[1], batch[2]
            decoder_attention, decoder_output_ids = batch[3], batch[4]

            inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                      'decoder_attention_mask': decoder_attention, 'lm_labels': decoder_output_ids, 'generate': False,
                      'partial_loss': args.partial_loss}

            outputs = model(**inputs)

            loss = outputs[0]

            tr_loss_sent += loss.item()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    logs = {}
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    loss_scalar_sent = (tr_loss_sent - logging_loss_sent) / args.save_steps
                    logs["loss_sent"] = loss_scalar_sent
                    logging_loss_sent = tr_loss_sent

                    # logs['loss_gen'] = tr_loss_gen

                    print(json.dumps({**logs, **{"step": global_step}}))
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # Evaluation
                    evaluate(args, eval_dataset, model, tokenizer)
                    save_checkpoints(args, args.output_dir, model, tokenizer, optimizer, scheduler)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        save_checkpoints(args, args.output_dir, model, tokenizer, optimizer, scheduler)
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    evaluate(args, eval_dataset, model, tokenizer)

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
        "--input_dir",
        default=None,
        type=str,
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
        required=False,
        help="Evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=False,
        help="The input training data file (a text file)."
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size evaluation.", )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="No. steps before backward pass.",)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs", )
    parser.add_argument("--max_steps", default=-1, type=int, help="If>0: no. train steps. Overrides num_train_epochs.",)
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=10, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--generate", action="store_true", help="Generate summaries for dev set", )
    parser.add_argument("--seed", type=int, default=100, help="random seed for initialization")
    parser.add_argument("--partial_loss", action="store_true", help="Use DAE-based partial loss scheme for training")

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

    args.n_gpu = 1
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

    eval_dataset = train_generation_utils.load_and_cache_examples_bart_partial(args, tokenizer, evaluate=True)
    evaluate(args, eval_dataset, model, tokenizer)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = train_generation_utils.load_and_cache_examples_bart_partial(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
