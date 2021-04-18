import torch
import logging
from torch import nn
import os
import csv
from torch.utils.data import TensorDataset
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
from torch.nn import CrossEntropyLoss
from pycorenlp import StanfordCoreNLP

logger = logging.getLogger(__name__)


def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def get_train_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "train.tsv"))


def get_dev_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "dev.tsv"))


def pad_1d(input, max_length, pad_token):
    padding_length = max_length - len(input)
    if padding_length < 0:
        input = input[:max_length]
        padding_length = 0
    input = input + ([pad_token] * padding_length)
    return input


def get_relevant_deps_and_context(line, nlp):
    dep_type = 'enhancedDependencies'
    ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']

    parse = nlp.annotate(line, properties={'annotators': 'tokenize,ssplit,pos,depparse', 'outputFormat': 'json',
                                           'ssplit.isOneSentence': True})
    deps = []
    tokens = parse['sentences'][0]['tokens']
    pos = [tok['pos'] for tok in tokens]
    tokens = [tok['word'] for tok in tokens]

    for dep_dict in parse['sentences'][0][dep_type]:

        if dep_dict['dep'] not in ignore_dep:
            dep_temp = {'dep': dep_dict['dep']}
            dep_temp.update({'child': dep_dict['dependentGloss'], 'child_idx': dep_dict['dependent']})
            dep_temp.update({'head': dep_dict['governorGloss'], 'head_idx': dep_dict['governor']})
            deps.append(dep_temp)
    return tokens, pos, deps


def get_tokenized_text(input_text, nlp):
    tokenized_json = nlp.annotate(input_text, properties={'annotators': 'tokenize', 'outputFormat': 'json',
                                                          'ssplit.isOneSentence': True})
    tokenized_text = []
    for tok in tokenized_json['tokens']:
        tokenized_text.append(tok['word'])
    tokenized_text = ' '.join(tokenized_text)
    return tokenized_text


def get_single_features(decode_text, input_text, tokenizer, nlp, args):
    inp_tok, inp_pos, input_dep = get_relevant_deps_and_context(decode_text, nlp)
    tokenized_text = get_tokenized_text(input_text, nlp)

    ex = {'input': tokenized_text, 'deps': [], 'context': ' '.join(inp_tok), 'sentlabel': 1}
    for dep in input_dep:
        ex['deps'].append({'dep': dep['dep'], 'label': 1,
                           'head_idx': dep['head_idx'] - 1, 'child_idx': dep['child_idx'] - 1,
                           'child': dep['child'], 'head': dep['head']})

    dict_temp = {'id': 0, 'input': ex['input'], 'sentlabel': ex['sentlabel'], 'context': ex['context']}
    for i in range(20):
        if i < len(ex['deps']):
            dep = ex['deps'][i]
            dict_temp['dep_idx' + str(i)] = str(dep['child_idx']) + ' ' + str(dep['head_idx'])
            dict_temp['dep_words' + str(i)] = str(dep['child']) + ' ' + str(dep['head'])
            dict_temp['dep' + str(i)] = dep['dep']
            dict_temp['dep_label' + str(i)] = dep['label']
        else:
            dict_temp['dep_idx' + str(i)] = ''
            dict_temp['dep_words' + str(i)] = ''
            dict_temp['dep' + str(i)] = ''
            dict_temp['dep_label' + str(i)] = ''

    features = convert_examples_to_features(
        [dict_temp],
        tokenizer,
        max_length=args.max_seq_length,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_attention_mask = torch.tensor([f.input_attention_mask for f in features], dtype=torch.long)

    child_indices = torch.tensor([f.child_indices for f in features], dtype=torch.long)
    head_indices = torch.tensor([f.head_indices for f in features], dtype=torch.long)

    mask_entail = torch.tensor([f.mask_entail for f in features], dtype=torch.long)
    mask_cont = torch.tensor([f.mask_cont for f in features], dtype=torch.long)
    num_dependencies = torch.tensor([f.num_dependencies for f in features], dtype=torch.long)
    arcs = torch.tensor([f.arcs for f in features], dtype=torch.long)

    sentence_label = torch.tensor([f.sentence_label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, input_attention_mask, child_indices, head_indices,
                            mask_entail, mask_cont, num_dependencies, arcs, sentence_label)

    return dataset


class ElectraBasicModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, attention, child, head, mask_entail, mask_cont,
                sent_label, num_dependency, device='cuda'):
        transformer_outputs = self.electra(input_ids, attention_mask=attention)
        output = transformer_outputs[0]

        output = self.dropout(output)
        output_pooled = output[:, 0]

        logits_all = self.classifier(output_pooled)

        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits_all.view(-1, 2), sent_label.view(-1))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return


class ElectraDependencyModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dep_label_classifier = nn.Linear(2 * config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, attention, child, head, mask_entail, mask_cont,
                sent_label, num_dependency, device='cuda'):
        batch_size = input_ids.size(0)

        transformer_outputs = self.electra(input_ids, attention_mask=attention)

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        outputs = outputs.view((-1, outputs.size(-1)))

        add = torch.arange(batch_size) * input_ids.size(1)
        add = add.unsqueeze(1).to(device)
        child_temp = child + add
        head_temp = head + add

        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]

        child_embeddings = child_embeddings.view(batch_size, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))

        final_embeddings = torch.cat([child_embeddings, head_embeddings], dim=2)
        logits_all = self.dep_label_classifier(final_embeddings)

        mask = torch.arange(mask_entail.size(1)).to(device)[None, :] >= num_dependency[:, None]
        mask = mask.type(torch.long) * -1
        mask_entail = mask_entail + mask
        mask_entail = mask_entail.detach()

        loss_fct_dep = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct_dep(logits_all.view(-1, 2), mask_entail.view(-1).type(torch.long))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return


class ElectraConstModelTwoClass(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dep_label_classifier = nn.Linear(2 * config.hidden_size, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def forward(self, input_ids, attention, child, head, mask_entail, mask_cont,
                sent_label, num_dependency, device='cuda'):
        batch_size = input_ids.size(0)

        transformer_outputs = self.electra(input_ids, attention_mask=attention)
        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        outputs = outputs.view((-1, outputs.size(-1)))

        add = torch.arange(batch_size) * input_ids.size(1)
        add = add.unsqueeze(1).to(device)
        child_temp = child + add
        head_temp = head + add

        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]

        child_embeddings = child_embeddings.view(batch_size, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))

        final_embeddings = torch.cat([child_embeddings, head_embeddings], dim=2)
        logits_all = self.dep_label_classifier(final_embeddings)
        log_softmax = self.logsoftmax(logits_all)[:, :, 1]

        log_likelihood_pos_labels = log_softmax * mask_entail.type(torch.float)
        log_likelihood_pos_labels = torch.sum(log_likelihood_pos_labels, dim=1)

        logprob_no_neg_labels = log_softmax * mask_cont.type(torch.float)
        logprob_no_neg_labels = torch.sum(logprob_no_neg_labels, dim=1)

        # zero = torch.tensor([0]).to(device).type(torch.float)
        # log_likelihood_one_neg_label = torch.min(torch.cat([torch.log(1 - torch.exp(logprob_no_neg_labels) + 1), zero]))
        log_likelihood_one_neg_label = torch.log(1 - torch.exp(logprob_no_neg_labels) + 1e-4)
        loss = torch.neg(torch.sum(log_likelihood_pos_labels + log_likelihood_one_neg_label))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return


class InputFeatures(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def convert_examples_to_features(examples, tokenizer, max_length=128, pad_token=None, num_deps_per_ex=20):
    features = []
    rejected_ex = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        tokens_input = []
        tokens_input.extend(tokenizer.tokenize('[CLS]'))
        index_now = len(tokens_input)
        for (word_index, word) in enumerate(example['input'].split(' ')):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_input.extend(word_tokens)
                index_now += len(word_tokens)

        tokens_input_more = []
        tokens_input_more.extend(tokenizer.tokenize('[SEP]'))
        index_now += 1

        index_map = {}
        for (word_index, word) in enumerate(example['context'].split(' ')):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens_input_more.extend(word_tokens)
                index_now += len(word_tokens)
                index_map[word_index] = index_now - 1
        tokens_input_more.extend(tokenizer.tokenize('[SEP]'))

        if len(tokens_input) + len(tokens_input_more) > max_length:
            extra_len = len(tokens_input) + len(tokens_input_more) - max_length + 2
            tokens_input = tokens_input[:-extra_len]
            for w in index_map:
                index_map[w] = index_map[w] - extra_len

        tokens_input = tokens_input + tokens_input_more

        child_indices = [0] * num_deps_per_ex
        head_indices = [0] * num_deps_per_ex

        mask_entail = [0] * num_deps_per_ex
        mask_cont = [0] * num_deps_per_ex
        num_dependencies = 0

        input_arcs = [[0] * 20 for _ in range(num_deps_per_ex)]
        sentence_label = int(example['sentlabel'])

        for i in range(num_deps_per_ex):
            if example['dep_idx' + str(i)] == '':
                break

            child_idx, head_idx = example['dep_idx' + str(i)].split(' ')
            child_idx = int(child_idx)
            head_idx = int(head_idx)

            num_dependencies += 1
            d_label = int(example['dep_label' + str(i)])

            if d_label == 1:
                mask_entail[i] = 1
            else:
                mask_cont[i] = 1

            child_indices[i] = index_map[child_idx]
            head_indices[i] = index_map[head_idx]

            w1 = example['dep_words' + str(i)].split(' ')[0]
            w2 = example['dep_words' + str(i)].split(' ')[1]
            arc_text = example['dep' + str(i)] + ' [SEP] ' + w1 + ' [SEP] ' + w2
            arc = tokenizer.encode(arc_text)
            input_arcs[i] = pad_1d(arc, 20, pad_token)

        if num_dependencies == 0:
            rejected_ex += 1
            continue

        # inputs = tokenizer.encode_plus(example['input'], example['context'], add_special_tokens=True)
        input_ids = tokenizer.convert_tokens_to_ids(tokens_input)

        assert len(tokens_input) == len(input_ids), 'length mismatched'
        padding_length_a = max_length - len(tokens_input)
        input_ids = input_ids + ([pad_token] * padding_length_a)
        input_attention_mask = [1] * len(tokens_input) + ([0] * padding_length_a)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_attention_mask=input_attention_mask,
                                      sentence_label=sentence_label,
                                      child_indices=child_indices,
                                      head_indices=head_indices,
                                      mask_entail=mask_entail,
                                      mask_cont=mask_cont,
                                      num_dependencies=num_dependencies,
                                      arcs=input_arcs))

    # print(rejected_ex)
    return features


def load_and_cache_examples(args, tokenizer, evaluate):
    if evaluate:
        data_dir = '/'.join(args.eval_data_file.split('/')[:-1])
    else:
        data_dir = '/'.join(args.train_data_file.split('/')[:-1])

    model_type = args.model_type

    if 'electra' in args.model_type:
        model_type = 'electra'

    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            model_type,
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = (get_dev_examples(data_dir) if evaluate else get_train_examples(data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_attention_mask = torch.tensor([f.input_attention_mask for f in features], dtype=torch.long)

    child_indices = torch.tensor([f.child_indices for f in features], dtype=torch.long)
    head_indices = torch.tensor([f.head_indices for f in features], dtype=torch.long)

    mask_entail = torch.tensor([f.mask_entail for f in features], dtype=torch.long)
    mask_cont = torch.tensor([f.mask_cont for f in features], dtype=torch.long)
    num_dependencies = torch.tensor([f.num_dependencies for f in features], dtype=torch.long)
    arcs = torch.tensor([f.arcs for f in features], dtype=torch.long)

    sentence_label = torch.tensor([f.sentence_label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, input_attention_mask, child_indices, head_indices,
                            mask_entail, mask_cont, num_dependencies, arcs, sentence_label)

    return dataset
