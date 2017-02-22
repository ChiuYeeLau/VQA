import json
import os
import argparse
import pdb

def main(params):
	train = []
	test = []
	imdir='v7w_%s.jpg'
	print 'Loading annotations and questions...'
	data = json.load(open('dataset_v7w_%s.json' %(params['data_set']), 'r'))["images"]
	# pdb.set_trace()

	for image in data:
		# print image.keys()
		for QA in image['qa_pairs']:
			correct_ans = QA['answer']
			question_id = QA['qa_id']
			image_path = imdir%(QA["image_id"])
			question = QA['question']
			# add correct answer
			train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': correct_ans, 'ans': 1})
			mc_ans = QA['multiple_choices']
			# add wrong answers
			for wrong_ans in mc_ans:
				train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': wrong_ans, 'ans': 0})

	# subtype = 'val2014'
	# for i in range(len(val_anno['annotations'])):
	#     ans = val_anno['annotations'][i]['multiple_choice_answer']
	#     question_id = val_anno['annotations'][i]['question_id']
	#     image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])

	#     question = val_ques['questions'][i]['question']
	#     mc_ans = val_ques['questions'][i]['multiple_choices']

	#     test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans})

	val_sz = len(data) // 5

	test = train[:val_sz]
	train = train[val_sz:]

	json.dump(train, open('vqa_raw_train.json', 'w'))
	json.dump(test, open('vqa_raw_test.json', 'w'))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_set', default = 'telling',help = 'which data set, telling or pointing')
	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	main(params)









