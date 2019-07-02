#!/bin/bash
##yelp_all
nohup python main.py /data/lengjia/topic_model/yelp_2013_39923/yelp/tf100/vocab_3000/ --train_data_file /data/lengjia/topic_model/yelp_2013_39923/yelp/yelp-2013-train.txt.ss --test_data_file /data/lengjia/topic_model/yelp_2013_39923/yelp/yelp-2013-test.txt.ss --num_label 5 --cuda 0 --maxmin 3 --batchsize 1 --temp_batch_num 10000 --dt 30 --vocab 3000 --clf_weight 9 --tm_weight 1 --max_epochs 20 --min_epochs 10 &
##imdb_all
#python main.py /data/lengjia/topic_model/imdb/tf100/vocab_3000/ --train_data_file /data/lengjia/topic_model/imdb/imdb_train.txt.ss --test_data_file /data/lengjia/topic_model/imdb/imdb_test.txt.ss --num_label 10 --cuda 2 --maxmin 3 --batchsize 1 --temp_batch_num 5000 --dt 30 --vocab 3000 --clf_weight 9 --tm_weight 1 --max_epochs 20 --min_epochs 10
