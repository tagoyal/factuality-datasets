import csv

input_file = open('$INPUT_FOLDER/dev_out.txt')

output_file = open('$OUTPUT_FOLDER/filename.tsv', 'w')

fieldnames = ['input', 'output', 'output_ids']
writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
writer.writeheader()


class Example(object):
    def __init__(self, inp):
        self.sent_text = inp[0]
        self.summary = inp[1]
        self.deps = []

        for idx in range(2, len(inp) - 2, 5):
            dep_ids = inp[idx + 1].split('\t')
            assert len(dep_ids) == 2, 'error'
            pred = int(inp[idx + 3].split('\t')[1])
            self.deps.append({'w1_idx': dep_ids[0], 'w2_idx': dep_ids[1], 'pred': pred})


def read_file(file):
    all_lines = []
    for line in file.readlines():
        if line.strip() == '':
            continue
        all_lines.append(line.strip())

    examples = []
    example_temp = []
    for line in all_lines:
        if line.startswith('sent_pred'):
            example_temp.append(line)
            examples.append(Example(example_temp))
            example_temp = []
        else:
            example_temp.append(line)

    return examples


examples = read_file(input_file)

for ex_idx, ex in enumerate(examples):
    article = ex.sent_text
    summary = ex.summary
    incorrect_ids = set([])
    for dep in ex.deps:
        if dep['pred'] == 0:
            incorrect_ids.add(dep['w1_idx'])
            incorrect_ids.add(dep['w2_idx'])

    incorrect_ids = ' '.join(list(incorrect_ids))
    dict_temp = {'input': article, 'output': summary, 'output_ids': incorrect_ids}
    writer.writerow(dict_temp)
