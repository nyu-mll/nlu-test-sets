import os
import sys
import numpy as np

np.random.seed(123456)

# ori_data contains original data
# data will store new custom split data
data_path = 'experiments/tasks/ori_data'
custom_data_path = 'experiments/tasks/data'

if not os.path.exists(custom_data_path):
	os.makedirs(custom_data_path)

task = sys.argv[1]

if task == 'winogrande':
	task = task + '/winogrande_1.1'
	
fnames = os.listdir(os.path.join(data_path, task))

dev_name = 'dev'
test_name = 'test'
dev_files = []
num_examples = 0
for fname in fnames:
	if fname.endswith('gz'):
		continue
	elif fname.startswith('valid'):
		dev_name = 'valid'
		dev_files.append(fname)
	elif fname.startswith('val'):
		dev_name = 'val'
		dev_files.append(fname)
	elif fname.startswith('dev'):
		dev_name = 'dev'
		dev_files.append(fname)
	elif fname.startswith('tests'):  # for piqa
		test_name = 'tests'
	elif fname.startswith('test'):
		test_name = 'test'

f = open(os.path.join(data_path, task, dev_files[0])).readlines()
num_examples = len(f)
dev_idx = np.random.choice(num_examples, num_examples // 2, replace=False)

print(dev_files)
print(dev_name, test_name)
for fname in dev_files:
	f = open(os.path.join(data_path, task, fname)).readlines()
	fdev = open(os.path.join(custom_data_path, task, fname), 'w')
	ftest = open(os.path.join(custom_data_path, task, fname.replace(dev_name, test_name)), 'w')

	print(len(f), num_examples)
	assert len(f) == num_examples
	for i in range(num_examples):
	        if i in dev_idx:
	            fdev.write(f[i])
	        else:
	            ftest.write(f[i])
	fdev.close()
	ftest.close()

print(task)
print('Dev examples', len(dev_idx))
print('Test examples', num_examples - len(dev_idx))
