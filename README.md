# A neural sequence-to-sequence parser for converting natural language queries to logical form

This is a tensorflow implementation of the sequence-to-sequence+attention parser model by Dong et al. (2016) described in the following paper.

''Language to Logical Form with Neural Attention'', Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016. https://arxiv.org/abs/1601.01280

Note that the sequence-to-tree+attention parser, also presented in the above paper, has not been implemented in this code. 

## Platform:

*Warning:* This implementation is based on an old Tensorflow version. This repo is no longer being maintained or updated.

- Tensorflow 1.0.0 
- Python 3.5

## Example usage:

### For training model:

```
python model/parse_s2s_att.py --data_dir=data --train_dir=checkpoint --train_file=geoqueries_train.txt --test_file=geoqueries_test.txt
```

### For testing model:

```
python model/parse_s2s_att.py --data_dir=data --train_dir=checkpoint --test_file=geoqueries_test.txt --test=True
```

### For interactive testing:

```
python model/parse_s2s_att.py --data_dir=data --train_dir=checkpoint --decode=True
```

The default parameters provided give test accuracy of 83.9% on the geo-queries dataset. However, this can vary slightly on different machines.


