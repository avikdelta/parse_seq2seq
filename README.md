A neural sequence-to-sequence parser for converting natural language queries to logical form.

This is a tenorflow implementation of the sequence-to-sequence+attention parser model by Dong et al. (2016) described in the following paper.

''Language to Logical Form with Neural Attention'', Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016. https://arxiv.org/abs/1601.01280

Example usage:

For training model:
 
python model/parse_s2s_att.py --data_dir=data --train_dir=checkpoint --train_file=geoqueries_train.txt --test_file=geoqueries_test.txt

For testing model:

python model/parse_s2s_att.py --data_dir=data --train_dir=checkpoint --test_file=geoqueries_test.txt --test=True

For interactive testing:

python model/parse_s2s_att.py --data_dir=data --train_dir=checkpoint --decode=True

The default parameters provided give test accuracy of 83.9% on the geo-queries dataset. However, this can vary slightly on different machines.


