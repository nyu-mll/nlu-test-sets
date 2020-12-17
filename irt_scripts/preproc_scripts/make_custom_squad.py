import os
import numpy as np
import json

np.random.seed(123456)

# ori_data contains original data, i.e., squad_v2 folder
# data will store new custom split data

source_dir = "experiments/tasks/ori_data"
target_dir = "experiments/tasks/data"
split = ['train', 'dev']

examples = []
num_qas = 0
for s in split:

	if s != 'dev':
		continue

	path = 'squad_v2/' + s + '-v2.0.json'
	paragraphs = []
	with open(os.path.join(source_dir, path)) as f:
		
		data = json.load(f)
		examples = data['data']

		num_examples = len(examples)
		dex_examples = []
		test_examples = []

		num_dev, num_test = 0, 0
		fdev = open(os.path.join(target_dir, path), 'w')
		ftest = open(os.path.join(target_dir, path.replace('dev', 'test')), 'w')
		dev_idx = np.random.choice(num_examples, num_examples // 2, replace=False)
		for i in range(num_examples):
			if i in dev_idx:
				dex_examples.append(examples[i])
				num_dev += 1
			else:
				test_examples.append(examples[i])
				num_test += 1
				

		dev_data = {"version": "v2.0", 
					"data": dex_examples}
		json.dump(dev_data, fdev, ensure_ascii=False)

		test_data = {"version": "v2.0", 
					"data": test_examples}
		json.dump(dev_data, ftest, ensure_ascii=False)

		fdev.close()
		ftest.close()

print('Dev', num_dev)
print('Test', num_test)
