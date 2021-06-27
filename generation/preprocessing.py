import argparse, csv
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
relevant_pos = ['VBD', 'VBN', 'VB', 'VBG', 'VBZ', 'NN', 'NNS']


def get_tokens(input_text):
    tokenized_json = nlp.annotate(input_text, properties={'annotators': 'tokenize', 'outputFormat': 'json',
                                                          'ssplit.isOneSentence': True})
    tokenized_text = []
    for tok in tokenized_json['tokens']:
        tokenized_text.append(tok['word'])
    tokenized_text = ' '.join(tokenized_text)
    return tokenized_text


def get_relevant_deps_and_context(line, args):
    dep_type = args.dependency_type
    parse = nlp.annotate(line, properties={'annotators': 'tokenize,ssplit,pos,depparse', 'outputFormat': 'json',
                                           'ssplit.isOneSentence': True})
    deps = []

    tokens = parse['sentences'][0]['tokens']
    tokens = [tok['word'] for tok in tokens]

    for dep_dict in parse['sentences'][0][dep_type]:
        if dep_dict['dep'] not in ignore_dep:
            dep_temp = {'dep': dep_dict['dep']}
            dep_temp.update({'child': dep_dict['dependentGloss'], 'child_idx': dep_dict['dependent']})
            dep_temp.update({'head': dep_dict['governorGloss'], 'head_idx': dep_dict['governor']})
            deps.append(dep_temp)
    return tokens, deps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data to train DAE and sentence level factuality models')
    parser.add_argument('--output_file', metavar='RESULTS_DIR',
                        default='./output_folder/train.tsv',
                        help='results dir')
    parser.add_argument('--input_file', metavar='INPUT_DIR',
                        default='./raw_data/train.tsv',
                        help='input dir')
    parser.add_argument("--dependency_type", default='enhancedDependencies', help='type of dependency labels')
    args = parser.parse_args()

    # generate and dump headers for output file
    output_file = open(args.output_file, 'w')
    fieldnames = ['id', 'input', 'sentlabel', 'context']
    for i in range(20):
        fieldnames.append('dep_idx' + str(i))
        fieldnames.append('dep_words' + str(i))
        fieldnames.append('dep' + str(i))
        fieldnames.append('dep_label' + str(i))

    writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    # read input file
    input_file = csv.DictReader(open(args.input_file))
    for idx, row in enumerate(input_file):
        article = input_file['article']
        summary = input_file['summary']

        article_text = get_tokens(article)
        toks, deps_list = get_relevant_deps_and_context(summary, args)

        sent_label = 0  # default to 0, not used

        ex = {'input': article_text, 'deps': [], 'context': ' '.join(toks), 'sentlabel': sent_label}
        for dep in deps_list:
            head_idx, child_idx = dep['head_idx'] - 1, dep['child_idx'] - 1
            dep_label = 0  # default to 0, not used

            ex['deps'].append({'dep': dep['dep'], 'label': dep_label,
                               'head_idx': head_idx, 'child_idx': child_idx,
                               'child': dep['child'], 'head': dep['head']})

        dict_temp = {'id': idx, 'input': ex['input'], 'sentlabel': ex['sentlabel'],
                     'context': ex['context']}

        for i, dep in enumerate(ex['deps']):
            if i >= 20:
                break
            dict_temp['dep_idx' + str(i)] = str(dep['child_idx']) + ' ' + str(dep['head_idx'])
            dict_temp['dep_words' + str(i)] = str(dep['child']) + ' ' + str(dep['head'])
            dict_temp['dep' + str(i)] = dep['dep']
            dict_temp['dep_label' + str(i)] = dep['label']

        writer.writerow(dict_temp)