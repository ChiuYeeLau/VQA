#!/usr/bin/env python
# encoding: utf-8

import json

test_ans_file = "vqa_raw_test.json"
test_pred_file = "data_90000.json"

with open(test_ans_file) as handle:
    ans = json.load(handle)

with open(test_pred_file) as handle:
    pred = json.load(handle)

accuracy = 0
for i in range(len(ans)):
    if ans[i]['ans'] == pred[i]['answer']:
        accuracy += 1.0

accuracy /= len(ans)

print "Achieved Accuracy:" + str(accuracy)
