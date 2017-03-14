#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from tensorflow.python.ops import rnn_cell
from sklearn.metrics import average_precision_score
import pdb
import spacy
from random import seed

seed(862)
tf.set_random_seed(862)

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, max_words_q, vocabulary_size, drop_out_rate, emb_matrix):
        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size
        self.drop_out_rate = drop_out_rate

        # Before-LSTM-embedding
        self.embed_ques_W = tf.Variable(emb_matrix, name='embed_ques_W')
        # self.embed_ques_W.assign(emb_matrix)

        '''
        # encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True,state_is_tuple=False)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True,state_is_tuple=False)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        # self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2],state_is_tuple=False)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1],state_is_tuple=False)
        '''

        # question-embedding W1
        self.embed_Q_W = tf.Variable(tf.random_uniform([self.input_embedding_size, self.dim_hidden], -0.08,0.08),name='embed_Q_W')
        self.embed_Q_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_Q_b')

        # Answer-embedding W3
        self.embed_A_W = tf.Variable(tf.random_uniform([self.input_embedding_size, self.dim_hidden], -0.08,0.08),name='embed_A_W')
        self.embed_A_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_A_b')

        # image-embedding W2
        self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_image_b')

        # score-embedding W4
        #self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        #self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')
        self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')

        # QI-embedding W3
        self.embed_QI_W = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.08, 0.08), name='embed_QI_W')
        self.embed_QI_b = tf.Variable(tf.random_uniform([dim_hidden], -0.08, 0.08), name='embed_QI_b')

    def build_model(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        answer = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        question_length = tf.placeholder(tf.int32, [self.batch_size])
        answer_length = tf.placeholder(tf.int32, [self.batch_size])
        label = tf.placeholder(tf.float32, [self.batch_size,2])

        state = tf.zeros([self.batch_size, self.input_embedding_size])
        state_que = tf.zeros([self.batch_size, self.input_embedding_size])  #zhe
        state_ans = tf.zeros([self.batch_size, self.input_embedding_size])  #zhe
        loss = 0.0
        question_ans = tf.concat(0, [question, answer])
        question_ans_length = tf.concat(0, [question_length, answer_length])
        q_a_length = tf.reshape(question_ans_length, [-1, 1])    # Convert to a len(yp) x 1 matrix.
        q_a_length = tf.tile(q_a_length, [1, self.input_embedding_size])  # Create multiple columns.
        q_a_length = tf.cast(q_a_length, tf.float32)

        # pdb.set_trace()

        inputs = tf.nn.embedding_lookup(self.embed_ques_W, question_ans)
        inputs = tf.div(tf.reduce_sum(inputs, 1), q_a_length)
        loss = 0.0
        state_que = inputs[0:500,:]
        state_ans = inputs[500:,:]

        '''
        for i in range(max_words_q):
            if i==0:
                blstm_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
                blstm_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question_ans[:,i-1])
            blstm_emb_drop = tf.nn.dropout(blstm_emb_linear, 1-self.drop_out_rate)
            blstm_emb = tf.tanh(blstm_emb_drop)

            output, state = self.stacked_lstm(blstm_emb, state)
            state_que = state[0:250,:]    #zhe
            state_ans = state[250:,:]  #zhe
        '''


        # multimodal (fusing question & image)
        Q_drop = tf.nn.dropout(state_que, 1-self.drop_out_rate)
        Q_linear = tf.nn.xw_plus_b(Q_drop, self.embed_Q_W, self.embed_Q_b)
        Q_emb = tf.tanh(Q_linear)

        image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
        image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_linear)

        A_drop = tf.nn.dropout(state_ans, 1-self.drop_out_rate)
        A_linear = tf.nn.xw_plus_b(A_drop, self.embed_A_W, self.embed_A_b)
        A_emb = tf.tanh(A_linear)

        QI = tf.mul(Q_emb, image_emb)

        QI_drop = tf.nn.dropout(QI, 1-self.drop_out_rate)
        QI_linear = tf.nn.xw_plus_b(QI_drop, self.embed_QI_W, self.embed_QI_b)
        QI_emb = tf.tanh(QI_linear)

        QIA = tf.mul(QI_emb, A_emb)
        scores_emb = tf.nn.xw_plus_b(QIA, self.embed_scor_W, self.embed_scor_b)   #zhe
        # Calculate cross entropy
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_emb, labels=label)   #zhe
        #scores = scores_emb#tf.transpose(scores_emb)
        #pdb.set_trace()
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scores_emb, labels=label)   #zhe
        # Calculate loss
        loss = tf.reduce_mean(cross_entropy)
        return loss, image, question, answer, question_length, answer_length, label

    def build_generator(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])
        answer = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])

        question_length = tf.placeholder(tf.int32, [self.batch_size])
        answer_length = tf.placeholder(tf.int32, [self.batch_size])

        state = tf.zeros([self.batch_size, self.input_embedding_size])
        state_que = tf.zeros([self.batch_size, self.input_embedding_size])  #zhe
        state_ans = tf.zeros([self.batch_size, self.input_embedding_size])  #zhe
        question_ans = tf.concat(0, [question, answer])
        question_ans_length = tf.concat(0, [question_length, answer_length])

        q_a_length = tf.reshape(question_ans_length, [-1, 1])    # Convert to a len(yp) x 1 matrix.
        q_a_length = tf.tile(q_a_length, [1, self.input_embedding_size])  # Create multiple columns.
        q_a_length = tf.cast(q_a_length, tf.float32)

        # pdb.set_trace()

        inputs = tf.nn.embedding_lookup(self.embed_ques_W, question_ans)
        inputs = tf.div(tf.reduce_sum(inputs, 1), q_a_length)

        loss = 0.0
        state_que = inputs[0:500,:]
        state_ans = inputs[500:,:]

        '''
        for i in range(max_words_q):
            if i==0:
                blstm_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
                blstm_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question_ans[:,i-1])
            blstm_emb_drop = tf.nn.dropout(blstm_emb_linear, 1-self.drop_out_rate)
            blstm_emb = tf.tanh(blstm_emb_drop)

            output, state = self.stacked_lstm(blstm_emb, state)
            state_que = state[0:250,:]    #zhe
            state_ans = state[250:,:]  #zhe
        '''

        tf.get_variable_scope().reuse_variables()

        # multimodal (fusing question & image)
        Q_drop = tf.nn.dropout(state_que, 1-self.drop_out_rate)
        Q_linear = tf.nn.xw_plus_b(Q_drop, self.embed_Q_W, self.embed_Q_b)
        Q_emb = tf.tanh(Q_linear)

        image_drop = tf.nn.dropout(image, 1-self.drop_out_rate)
        image_linear = tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_linear)

        A_drop = tf.nn.dropout(state_ans, 1-self.drop_out_rate)
        A_linear = tf.nn.xw_plus_b(A_drop, self.embed_A_W, self.embed_A_b)
        A_emb = tf.tanh(A_linear)

        QI = tf.mul(Q_emb, image_emb)

        QI_drop = tf.nn.dropout(QI, 1-self.drop_out_rate)
        QI_linear = tf.nn.xw_plus_b(QI_drop, self.embed_QI_W, self.embed_QI_b)
        QI_emb = tf.tanh(QI_linear)

        QIA = tf.mul(QI_emb, A_emb)
        scores_emb = tf.nn.xw_plus_b(QIA, self.embed_scor_W, self.embed_scor_b)   #zhe
        # Calculate cross entropy
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_emb, labels=label)   #zhe
        generated_ANS = tf.transpose(scores_emb)

        return generated_ANS, image, question, answer, question_length, answer_length

#####################################################
#                 Global Parameters         #
#####################################################
print('Loading parameters ...')
# Data input setting
input_img_h5 = './data_img_resnet_wikiGlove.h5'
input_ques_h5 = './data_prepro_wikiGlove.h5'
input_json = './data_prepro_wikiGlove.json'

# Train Parameters setting
learning_rate_global = 0.0003          # learning rate for rmsprop
learning_rate_nlp = 0.00075
learning_rate_decay_start = -1      # at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 500            # batch_size for each iterations
input_embedding_size = 300      # The encoding size of each token in the vocabulary
rnn_size = 512              # size of the rnn in number of hidden nodes in each layer
rnn_layer = 2               # number of the rnn layer
dim_image = 2048
dim_hidden = 2048 #1024         # size of the common embedding vector
num_output = 2#1000         # number of output answers
img_norm = 1                # normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083

# Check point
checkpoint_path = 'model_save/'

# misc
gpu_id = 0
max_itr = 150000
n_epochs = 300
max_words_q = 26
num_answer = 1000

f1 = open("RecordLoss.txt", "w")
f2 = open("TestAcurracies.txt", "w")
#####################################################

def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
    return v

def get_data():

    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]
    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------  at most 47000
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
        # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer
        tem = hf.get('ans_train')
        train_data['answer'] = np.array(tem)-1

        tem = hf.get('ans_length_train')
        train_data['length_a'] = np.array(tem)

        tem = hf.get('target_train')
        train_data['target'] = np.transpose(np.vstack((np.array(tem), 1-np.array(tem))))

        train_data['emb_matrix'] = np.array(hf.get('emb_matrix'))


    print('question & answer aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])
    train_data['answer'] = right_align(train_data['answer'], train_data['length_a'])


    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(2048,1))))



    return dataset, img_feature, train_data

def get_nonzero_num(np_arr):
    return (np_arr != 0).sum(1)

def get_data_test():

    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]
    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        # answer
        tem = hf.get('ans_test')
        test_data['answer'] = np.array(tem)-1

        tem = hf.get('ans_length_test')
        test_data['length_a'] = np.array(tem)

        tem = hf.get('target_test')
        test_data['target'] = np.transpose(np.vstack((np.array(tem), 1-np.array(tem))))

        test_data['emb_matrix'] = np.array(hf.get('emb_matrix'))

    print('question aligning')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])
    test_data['answer'] = right_align(test_data['answer'], test_data['length_a'])

    # pdb.set_trace()

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(2048,1))))

    return dataset, img_feature, test_data

def train():
    print('loading dataset...')
    dataset, img_feature_train, train_data = get_data()
    dataset_test, img_feature_test, test_data = get_data_test()
    num_train = train_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print('vocabulary_size : ' + str(vocabulary_size))

    print('constructing  model...')
    # pdb.set_trace()
    model = Answer_Generator(
        rnn_size = rnn_size,
        rnn_layer = rnn_layer,
        batch_size = batch_size,
        input_embedding_size = input_embedding_size,
        dim_image = dim_image,
        dim_hidden = dim_hidden,
        max_words_q = max_words_q,
        vocabulary_size = vocabulary_size,
        drop_out_rate = 0.5,
        emb_matrix = train_data['emb_matrix'])

    tf_loss, tf_image, tf_question, tf_answer, tf_question_length, tf_answer_length, tf_label = model.build_model()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=100)

    tvars = tf.trainable_variables()
    nlp_vars = [tvars[0]]
    global_vars = tvars[1:]
    lr_global = tf.Variable(learning_rate_global)
    lr_nlp = tf.Variable(learning_rate_nlp)
    opt_global = tf.train.AdamOptimizer(learning_rate = lr_global)
    opt_nlp = tf.train.AdamOptimizer(learning_rate = lr_nlp)

    # gradient clipping
    gvs = tf.gradients(tf_loss, nlp_vars + global_vars)
    gvs_nlp = [gvs[0]]
    gvs_global = gvs[1:]
    # clipped_gvs = [(tf.clip_by_value(grad, -500.0, 500.0), var) for grad, var in gvs]  ## either 100 or 10000 will result in Nan, original is 100
    train_op_nlp = opt_nlp.apply_gradients(zip(gvs_nlp, nlp_vars))
    train_op_global = opt_global.apply_gradients(zip(gvs_global, global_vars))
    train_op = tf.group(train_op_nlp, train_op_global)

    tf.initialize_all_variables().run()

    print('start training...')
    for itr in range(max_itr):
        tStart = time.time()
        # shuffle the training data
        index = np.random.random_integers(0, num_train-1, batch_size)

        # pdb.set_trace()

        current_question = train_data['question'][index,:]
        current_length_q = train_data['length_q'][index]
        current_answer = train_data['answer'][index]
        current_length_a = train_data['length_a'][index]
        current_img_list = train_data['img_list'][index]
        current_target = train_data['target'][index]
        current_img = img_feature_train[current_img_list,:]


        # do the training process!!!
        _, loss = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_image: current_img,
                        tf_question: current_question,
                        tf_answer: current_answer,
                        tf_label: current_target,
                        tf_question_length: current_length_q,
                        tf_answer_length: current_length_a
                        })

        current_learning_rate = lr*decay_factor
        lr.assign(current_learning_rate).eval()

        tStop = time.time()
        if np.mod(itr, 100) == 0:
            print ("Iteration: ", itr, " Loss: ", loss, " Learning Rate: ", lr.eval())
            f1.write(str(itr) + '\t' + str(loss) + "\n")
            #print ("Iteration: ", itr, " scores: ", scores, " label: ", current_target)
            print ("Time Cost:", round(tStop - tStart,2), "s")
        if np.mod(itr, 1800) == 0:
            print ("Iteration ", itr, " is done. Saving the model ...")
            saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)
            ## print test result
            '''
            model_name = 'model_save/model-' + str(itr)
            test(model_path = model_name)
            '''

            num_test = test_data['question'].shape[0]
            # print('numtest: ' + str(num_test))
            vocabulary_size = len(dataset_test['ix_to_word'].keys())

            tf_proba_test, tf_image_test, tf_question_test, tf_answer_test, tf_question_test_length, tf_answer_test_length = model.build_generator()

            result = {}

            for current_batch_start_idx in xrange(0, num_test-1, batch_size):
            #for current_batch_start_idx in xrange(0,3,batch_size):
                tStart = time.time()
                # set data into current*
                if current_batch_start_idx + batch_size < num_test:
                    current_batch_file_idx = range(current_batch_start_idx, current_batch_start_idx + batch_size)
                else:
                    current_batch_file_idx = range(current_batch_start_idx, num_test)

                current_question = test_data['question'][current_batch_file_idx,:]
                current_length_q = test_data['length_q'][current_batch_file_idx]
                current_img_list = test_data['img_list'][current_batch_file_idx]
                current_answer = test_data['answer'][current_batch_file_idx,:]
                current_length_a = test_data['length_a'][current_batch_file_idx]
                current_ques_id  = test_data['ques_id'][current_batch_file_idx]
                current_target = test_data['target'][current_batch_file_idx]
                current_img = img_feature_test[current_img_list,:] # (batch_size, dim_image)

                # deal with the last batch
                if(len(current_img)<500):
                        pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                        pad_q = np.zeros((500-len(current_img),max_words_q),dtype=np.int)
                        pad_q_len = np.zeros(500-len(current_length_q),dtype=np.int)
                        pad_q_id = np.zeros(500-len(current_length_q),dtype=np.int)
                        pad_img_list = np.zeros(500-len(current_length_q),dtype=np.int)
                        pad_a = np.zeros((500-len(current_img),max_words_q),dtype=np.int)
                        pad_a_len = np.zeros(500-len(current_length_a),dtype=np.int)
                        pad_target = np.zeros((500-len(current_target), 2),dtype=np.int)
                        current_img = np.concatenate((current_img, pad_img))
                        current_question = np.concatenate((current_question, pad_q))
                        current_length_q = np.concatenate((current_length_q, pad_q_len))
                        current_ques_id = np.concatenate((current_ques_id, pad_q_id))
                        current_img_list = np.concatenate((current_img_list, pad_img_list))
                        current_answer = np.concatenate((current_answer, pad_a))
                        current_length_a = np.concatenate((current_length_a, pad_a_len))
                        current_target = np.concatenate((current_target, pad_target))

                pred_proba = sess.run(
                        tf_proba_test,
                        feed_dict={
                            tf_image_test: current_img,
                            tf_question_test: current_question,
                            tf_answer_test: current_answer,
                            tf_question_test_length: current_length_q,
                            tf_answer_test_length: current_length_a
                            })

                # initialize json list
                pred_proba = np.transpose(pred_proba)
                assert(current_target.shape == (500,2))
                assert(pred_proba.shape == (500,2))

                target, prob = getMaximumLikelihood(current_target, pred_proba)

                for i in list(range(0, 500)):
                    if str(current_ques_id[i]) not in result:
                        result[str(current_ques_id[i])] = [target[i], prob[i]]
                    else:
                        if result[str(current_ques_id[i])][1] < prob[i]:
                            result[str(current_ques_id[i])] = [target[i], prob[i]]

                tStop = time.time()
                # print ("Testing batch: ", current_batch_file_idx[0])
                # print ("Time Cost:", round(tStop - tStart,2), "s")

            print ("Testing done.")
            tStop_total = time.time()
            # print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
            # Save to JSON
            # print ('Saving result...')
            acc = 0
            for k,v in result.iteritems():
                acc += v[0]
            print('Accuracy of test: ' + str(acc*1.0/len(result)))
            f2 = open("TestAcurracies.txt", "a")
            f2.write(str(itr) + '\t' + str(acc*1.0/len(result)) + "\n")
            f2.close()
            dd = json.dump(result,open('data.json','w'))




    print ("Finally, saving the model ...")
    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    f1.close()
    print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")

def test(model_path='model_save/model-5400'):
    # print ('loading dataset...')
    dataset, img_feature, test_data = get_data_test()
    num_test = test_data['question'].shape[0]

    # print('numtest: ' + str(num_test))

    vocabulary_size = len(dataset['ix_to_word'].keys())
    # print ('vocabulary_size : ' + str(vocabulary_size))

    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = batch_size,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            max_words_q = max_words_q,
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0,
            emb_matrix = test_data['emb_matrix'])

    tf_proba, tf_image, tf_question, tf_answer = model.build_generator()

    #sess = tf.InteractiveSession()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    tStart_total = time.time()
    result = {}

    for current_batch_start_idx in xrange(0, num_test-1, batch_size/2):
    #for current_batch_start_idx in xrange(0,3,batch_size):
        tStart = time.time()
        # set data into current*
        if current_batch_start_idx + batch_size/2 < num_test:
            current_batch_file_idx = range(current_batch_start_idx, current_batch_start_idx + batch_size/2)
        else:
            current_batch_file_idx = range(current_batch_start_idx, num_test)

        current_question = test_data['question'][current_batch_file_idx,:]
        current_length_q = test_data['length_q'][current_batch_file_idx]
        current_img_list = test_data['img_list'][current_batch_file_idx]
        current_answer = test_data['answer'][current_batch_file_idx,:]
        current_length_a = test_data['length_a'][current_batch_file_idx]
        current_ques_id  = test_data['ques_id'][current_batch_file_idx]
        current_target = test_data['target'][current_batch_file_idx]
        current_img = img_feature[current_img_list,:] # (batch_size, dim_image)

        # deal with the last batch
        if(len(current_img)<250):
                pad_img = np.zeros((250-len(current_img),dim_image),dtype=np.int)
                pad_q = np.zeros((250-len(current_img),max_words_q),dtype=np.int)
                pad_q_len = np.zeros(250-len(current_length_q),dtype=np.int)
                pad_q_id = np.zeros(250-len(current_length_q),dtype=np.int)
                pad_img_list = np.zeros(250-len(current_length_q),dtype=np.int)
                pad_a = np.zeros((250-len(current_img),max_words_q),dtype=np.int)
                pad_a_len = np.zeros(250-len(current_length_a),dtype=np.int)
                pad_target = np.zeros((250-len(current_target), 2),dtype=np.int)
                current_img = np.concatenate((current_img, pad_img))
                current_question = np.concatenate((current_question, pad_q))
                current_length_q = np.concatenate((current_length_q, pad_q_len))
                current_ques_id = np.concatenate((current_ques_id, pad_q_id))
                current_img_list = np.concatenate((current_img_list, pad_img_list))
                current_answer = np.concatenate((current_answer, pad_a))
                current_length_a = np.concatenate((current_length_a, pad_a_len))
                current_target = np.concatenate((current_target, pad_target))

        pred_proba = sess.run(
                tf_proba,
                feed_dict={
                    tf_image: current_img,
                    tf_question: current_question,
                    tf_answer: current_answer
                    })

        # initialize json list
        pred_proba = np.transpose(pred_proba)
        assert(current_target.shape == (250,2))
        assert(pred_proba.shape == (250,2))

        target, prob = getMaximumLikelihood(current_target, pred_proba)

        for i in list(range(0, 250)):
            if str(current_ques_id[i]) not in result:
                result[str(current_ques_id[i])] = [target[i], prob[i]]
            else:
                if result[str(current_ques_id[i])][1] < prob[i]:
                    result[str(current_ques_id[i])] = [target[i], prob[i]]

        tStop = time.time()
        # print ("Testing batch: ", current_batch_file_idx[0])
        # print ("Time Cost:", round(tStop - tStart,2), "s")

    print ("Testing done.")
    tStop_total = time.time()
    # print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
    # Save to JSON
    # print ('Saving result...')
    acc = 0
    for k,v in result.iteritems():
        acc += v[0]
    print('Accuracy of test: ' + str(acc*1.0/len(result)))
    dd = json.dump(result,open('data.json','w'))

def getMaximumLikelihood(raw_target, raw_prob):
    target = np.zeros((500,))
    prob = np.zeros((500,))
    for i in list(range(0, 500)):
        prob[i] = softmax(raw_prob[i,0], raw_prob[i,1])
        target[i] = raw_target[i,0]

    return target, prob

def softmax(a, b):
    return np.exp(a)/(np.exp(a) + np.exp(b))

if __name__ == '__main__':
    with tf.device('/gpu:'+str(0)):
        train()
    with tf.device('/gpu:'+str(1)):
        test()
