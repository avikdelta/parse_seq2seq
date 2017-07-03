# author: Avik Ray (avik.r@samsung.com) 
#
# script modified from tensorflow ENG->FR machine translation using 
# sequence-to-sequence tutorial code by The TensorFlow Authors
# 
# ========================================================================
"""Binary for training semantic parser model and decoding from them.

example usage: 

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model
from accuracy import *


tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.98,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.5,
                          "dropout rate.")
tf.app.flags.DEFINE_integer("batch_size", 20,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 2000, "Max natural language input vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 2000, "Max logical form vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training/Checkpoint directory.")
tf.app.flags.DEFINE_string("train_file", None, "Training data.")
tf.app.flags.DEFINE_string("dev_file", None, "Validation data.")
tf.app.flags.DEFINE_string("test_file", None, "Test data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("display_every", 100,
                            "How many training steps/batches to do per display output.")
tf.app.flags.DEFINE_integer("max_epoch", 90,
                            "maximum number of epochs to train.")
tf.app.flags.DEFINE_integer("eval_per_epoch", 10,
                            "evaluation frequency")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Set to True for computing test accuracy.")
tf.app.flags.DEFINE_boolean("valid", True,
                            "True = Use valid set for evaluation. False = Use test set for evaluation.")
tf.app.flags.DEFINE_boolean("display", False,
                            "Display inference/test output.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")



FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(50, 130)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only, use_dropout):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dropout=FLAGS.dropout,
      use_dropout=use_dropout,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if FLAGS.decode or FLAGS.test:
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model

#-----------------------------------------------------
# creates batches from the training data for training
#-----------------------------------------------------
def create_batches(data):
    print("generating batches...")
    batches = [[] for _ in _buckets]
    for bucket_id in xrange(len(_buckets)):
        data_bucket = data[bucket_id]
        encoder_size, decoder_size = _buckets[bucket_id]
        
        # shuffle the data
        data_permute = np.random.permutation(len(data_bucket))
        
        num_batches = math.ceil(len(data_bucket)/FLAGS.batch_size)
        for b_idx in xrange(num_batches):
            encoder_inputs, decoder_inputs = [], []
            for i in xrange(FLAGS.batch_size):
                data_idx = data_permute[(b_idx*FLAGS.batch_size+i) % len(data_bucket)]
                encoder_input, decoder_input = data_bucket[data_idx]

                # Encoder inputs are padded and then reversed.
                encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
                encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

                # Decoder inputs get an extra "GO" symbol, and are padded then.
                decoder_pad_size = decoder_size - len(decoder_input) - 1
                decoder_inputs.append([data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * decoder_pad_size)
                
            # Now we create batch-major vectors from the data selected above.
            batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

            # Batch encoder inputs are just re-indexed encoder_inputs.
            for length_idx in xrange(encoder_size):
                batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] 
                                            for batch_idx in xrange(FLAGS.batch_size)], dtype=np.int32))

            # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
            for length_idx in xrange(decoder_size):
                batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                                            for batch_idx in xrange(FLAGS.batch_size)], dtype=np.int32))

                # Create target_weights to be 0 for targets that are padding.
                batch_weight = np.ones(FLAGS.batch_size, dtype=np.float32)
                for batch_idx in xrange(FLAGS.batch_size):
                    # We set weight to 0 if the corresponding target is a PAD symbol.
                    # The corresponding target is decoder_input shifted by 1 forward.
                    if length_idx < decoder_size - 1:
                        target = decoder_inputs[batch_idx][length_idx + 1]
                    if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                        batch_weight[batch_idx] = 0.0
                        
                batch_weights.append(batch_weight)
                
            batches[bucket_id].append((batch_encoder_inputs, batch_decoder_inputs, batch_weights))
            
    return batches
    
#-----------------------------------------------------
# main training function
#-----------------------------------------------------
def train():
  """Train a query->logical form semantic parser model"""
  from_train = None
  to_train = None
  from_dev = None
  to_dev = None
  
  # process training data
  from_train_data = os.path.join(FLAGS.data_dir,"train_q.txt")
  to_train_data = os.path.join(FLAGS.data_dir,"train_f.txt")
  
  if FLAGS.train_file:
    print("using training files",from_train_data,"and",to_train_data)
    from_dev_data = from_train_data
    to_dev_data = to_train_data
    data_utils.splitToFrom(FLAGS.data_dir,FLAGS.train_file,"train") # split to-from data
    if FLAGS.dev_file:
      data_utils.splitToFrom(FLAGS.data_dir,FLAGS.dev_file,"valid") # split to-from data
      from_dev_data = os.path.join(FLAGS.data_dir,"valid_q.txt")
      to_dev_data = os.path.join(FLAGS.data_dir,"valid_f.txt")
      print("using validation files",from_dev_data,"and",to_dev_data,"for validation")
    elif FLAGS.test_file:
      data_utils.splitToFrom(FLAGS.data_dir,FLAGS.test_file,"test") # split to-from data
      from_dev_data = os.path.join(FLAGS.data_dir,"test_q.txt")
      to_dev_data = os.path.join(FLAGS.data_dir,"test_f.txt")
      print("using test files",from_dev_data,"and",to_dev_data,"for validation")
    else:
      print("using train files",from_dev_data,"and",to_dev_data,"for validation")
    
  
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        from_dev_data,
        to_dev_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)
  else:
      # Prepare data.
      print("Preparing data in %s" % FLAGS.data_dir)
      
  print("data preparation complete!")
  
  config_ = tf.ConfigProto()
  config_.gpu_options.allow_growth = True
  config_.allow_soft_placement = True
  with tf.Session(config=config_) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False, True)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(from_dev, to_dev)
    train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    print("train_bucket_sizes =",train_bucket_sizes)
    train_total_size = float(sum(train_bucket_sizes))
    print("train_total_size =",train_total_size)
                           
    # create data batches for training
    all_train_batches = create_batches(train_set)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    epoch_count = 1
    epoch_size = math.ceil(train_total_size/model.batch_size)
    previous_losses = []
    print("------------------------------------------")
    print("Starting training: max epochs =",FLAGS.max_epoch,"; epoch size =",epoch_size,"; train_total_size =",train_total_size)
    print("------------------------------------------")
    
    # generate random permutation of batch numbers for epoch
    batch_permutations = [np.random.permutation(len(all_train_batches[bkt_idx])) for bkt_idx in xrange(len(_buckets))]
    batch_pidx = [0 for bkt_idx in xrange(len(_buckets))]
    
    while True:
      
      bucket_id = 0 # we just have one bucket

      # Get a batch and make a step.
        
      # select batch for this iteration
      bidx = batch_permutations[bucket_id][batch_pidx[bucket_id]]
      encoder_inputs, decoder_inputs, target_weights = all_train_batches[bucket_id][bidx]
      batch_pidx[bucket_id] = (batch_pidx[bucket_id] + 1) % len(all_train_batches[bucket_id])
      
      # execute gradient descent step
      start_time = time.time()
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.display_every
      loss += step_loss / FLAGS.display_every
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.display_every == 0:
        # Print statistics for the previous epoch.
        print ("epoch %d/%d step %d learning rate %.6f time/batch %.2f loss "
               "%2.5f" % (epoch_count,FLAGS.max_epoch,model.global_step.eval(), model.learning_rate.eval(),
                         step_time, loss))
          
        previous_losses.append(loss)
        
        # zero timer and loss.
        step_time, loss = 0.0, 0.0
        sys.stdout.flush()
        
      # all epoch updates
      if current_step%epoch_size==0:
        epoch_count += 1
        
        # generate random permutation of batch numbers for epoch
        batch_permutations = [np.random.permutation(len(all_train_batches[bkt_idx])) for bkt_idx in xrange(len(_buckets))]
        batch_pidx = [0 for bkt_idx in xrange(len(_buckets))]
    
        # Decrease learning rate
        if epoch_count>5:
            sess.run(model.learning_rate_decay_op)
        
        # Run evals on development set and print their perplexity.
        if (epoch_count-1)%FLAGS.eval_per_epoch==0 or epoch_count>FLAGS.max_epoch:
            
            for bucket_id in xrange(len(_buckets)):
                if len(dev_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % (bucket_id))
                    continue
            
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    dev_set, bucket_id)
                _, eval_loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
                eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                print("  epoch %d eval: bucket %d perplexity %.2f loss %2.5f" % (epoch_count-1,bucket_id, eval_ppx, eval_loss))
        
        # save model at the end of last training epoch
        if epoch_count > FLAGS.max_epoch:
            print("Max epoch reached!")
            # Save checkpoint 
            checkpoint_path = os.path.join(FLAGS.train_dir, "parse.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print("Model saved.")
            break
            
    print("Training complete!")
    return

#-----------------------------------------------------
# function computing test accuracy
#-----------------------------------------------------
def test_accuracy(from_data,to_data):

    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    config_.allow_soft_placement = True
    with tf.Session(config=config_) as sess:
        # Create model and load parameters.
        model = create_model(sess, True, False)
        model.batch_size = 1  # We decode one sentence at a time.

        print("loading data...")
        # Load vocabularies.
        from_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.from" % FLAGS.from_vocab_size)
        to_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.to" % FLAGS.to_vocab_size)
        from_vocab, _ = data_utils.initialize_vocabulary(from_vocab_path)
        to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)

        # read test data
        test_data = data_utils.tokenize_dataset(from_data,to_data,from_vocab,to_vocab)
        print("data loaded. computing accuracy...")
        val_acc = 0
        comm_dict = init_comm_dict(to_vocab)
        rev_to_vocab_dict = reverseDict(to_vocab)
        
        for data in test_data:

            from_token_ids = data[0]
            to_token_ids = data[1]
      
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(from_token_ids):
                    bucket_id = i
                    break
                else:
                    logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(from_token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            
            outputs = post_process(outputs,to_vocab)
            
            if compute_tree_accuracy(outputs,to_token_ids,to_vocab,rev_to_vocab_dict,comm_dict,FLAGS.display):
                val_acc+= 1
                
            
        val_acc = val_acc/float(len(test_data))
        
    return val_acc
            

#-----------------------------------------------------
# function for interactive decoding
#-----------------------------------------------------
def decode():
  config_ = tf.ConfigProto()
  config_.gpu_options.allow_growth = True
  config_.allow_soft_placement = True
  with tf.Session(config=config_) as sess:
    # Create model and load parameters.
    model = create_model(sess, True, False)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    from_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    to_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.to" % FLAGS.to_vocab_size)
    from_vocab, _ = data_utils.initialize_vocabulary(from_vocab_path)
    _, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(sentence, from_vocab)
      
      print(token_ids)
      
      # Which bucket does it belong to?
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence)

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out logical form corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_to_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  config_ = tf.ConfigProto()
  config_.gpu_options.allow_growth = True
  config_.allow_soft_placement = True
  with tf.Session(config=config_) as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  elif FLAGS.test:
    from_test_data = os.path.join(FLAGS.data_dir,"test_q.txt")
    to_test_data = os.path.join(FLAGS.data_dir,"test_f.txt")
    if not (os.path.exists(from_test_data) and os.path.exists(to_test_data)):
        if FLAGS.test_file:
            data_utils.splitToFrom(FLAGS.data_dir,FLAGS.test_file,"test") # split to-from data
        else:
            print("test data file missing!")
            return
    
    print("computing test accuracy...")
    test_acc = test_accuracy(from_test_data,to_test_data)
    print("test accuracy =",test_acc)
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
