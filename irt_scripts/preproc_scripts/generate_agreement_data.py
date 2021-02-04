import os
import json

from collections import defaultdict, Counter

# script to breakdown MNLI/SNLI responses into agreement responses
# replace mnli_mismatched with mnli/snli as needed

# path to preprocessed data by jiant
jiant_file = "data/mnli_mismatched/test.jsonl"
# path to original distributed data
ori_file = "ori_data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"

ex2id = {}
id2ex = {}
duplicates = []
with open(jiant_file) as f:
	idx = 0
	for i, line in enumerate(f):
		data = json.loads(line)
		s1 = data['premise']
		s2 = data['hypothesis']
		if data['label'] == -1:
			continue

		if (s1, s2) in ex2id:
			duplicates.append((s1, s2))
			s1 = s1 + '*'
		ex2id[(s1, s2)] = idx
		id2ex[idx] = (s1, s2)
		idx += 1

ex2ann = {}
with open(ori_file) as f:
	for idx, line in enumerate(f):
		data = json.loads(line)
		s1 = data['sentence1']
		s2 = data['sentence2']
		if (s1, s2) in ex2id:
			ex2ann[(s1, s2)] = data['annotator_labels']
			if (s1, s2) in duplicates:
				s = s1 + '*'
				ex2ann[(s, s2)] = ex2ann[(s1, s2)]


agree3, agree4, agree5 = [], [], []
for idx in id2ex:
	ex = id2ex[idx]
	ann = ex2ann[ex]
	counts = Counter(ann)

	for k in counts.keys():
		if counts[k] == 3:
			agree3.append(idx)
		elif counts[k] == 4:
			agree4.append(idx)
		elif counts[k] == 5:
			agree5.append(idx)

print('Total examples:', len(agree3) + len(agree4) + len(agree5))
print(len(agree3))
print(len(agree4))
print(len(agree5))

irt_file = "nli-data/mnli_mismatched_irt_all_coded.csv"
f3 = open("nli-data/mnli_mismatched3_irt_all_coded.csv", "w")
f4 = open("nli-data/mnli_mismatched4_irt_all_coded.csv", "w")
f5 = open("nli-data/mnli_mismatched5_irt_all_coded.csv", "w")

with open(irt_file) as f:
	for idx, line in enumerate(f):

		items = line.split(',')
		m, responses = items[0], items[1:]

		assert len(responses) == len(ex2id)
		
		responses3 = [m] + [responses[i] for i in agree3]
		responses4 = [m] + [responses[i] for i in agree4]
		responses5 = [m] + [responses[i] for i in agree5]

		f3.write(','.join(responses3) + '\n')
		f4.write(','.join(responses4) + '\n')
		f5.write(','.join(responses5) + '\n')

f3.close()
f4.close()
f5.close()

