from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import requests
import os
import sys

import math
import numpy as np
import scipy.sparse
import cntk as C

import re
import string
import time


#C.device.try_set_default_device(C.device.cpu())

regex = re.compile(r"^[^\d\W]\w*$", re.UNICODE)
keywords = ["async", "await", "break", "continue", "class", "extends", "constructor", "super", "extends", "const", "let", "var", "debugger", "delete", "do", "while", "export", "import", "for", "each", "in", "of", "function", "return", "get", "set", "if", "else", "instanceof", "typeof", "null", "undefined", "switch", "case", "default", "this", "true", "false", "try", "catch", "finally", "void", "yield", "any", "boolean", "null", "never", "number", "string", "symbol", "undefined", "void", "as", "is", "enum", "type", "interface", "abstract", "implements", "static", "readonly", "private", "protected", "public", "declare", "module", "namespace", "require", "from", "of", "package"]

restore = False
train = 'seq2seq-train.ctf'
valid = 'seq2seq-valid.ctf'
test = 'seq2seq-test.ctf'
files = {
  'train': { 'file': train, 'location': 0 },
  'valid': { 'file': valid, 'location': 0 },
  'test': { 'file': test, 'location': 0 },
  'source': { 'file': 'source.wl', 'location': 1 },
  'target': { 'file': 'target.wl', 'location': 1 }
}

# load dictionaries
source_wl = [line.rstrip('\n') for line in open(files['source']['file'])]
target_wl = [line.rstrip('\n') for line in open(files['target']['file'])]
source_dict = {source_wl[i]:i for i in range(len(source_wl))}
target_dict = {target_wl[i]:i for i in range(len(target_wl))}

# number of words in vocab, slot labels, and intent labels
vocab_size = len(source_dict)
num_labels = len(target_dict)
print_freq = 10 # Number of steps per epoch, 
epoch_size = 4293536//print_freq # Total #tokens, tentatively //print_freq for increased print frequency
minibatch_size = 5000
minibatch_size = 1000
emb_dim    = 300
hidden_dim = 600
num_epochs = 10*print_freq

# Create the containers for input feature (x) and the label (y)
x = C.sequence.input_variable(vocab_size, name="x")
y = C.sequence.input_variable(num_labels, name="y")
t = C.sequence.input_variable(hidden_dim, name="t")
m = C.sequence.input_variable(1, name="m")

def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    G = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), G(x))
    return apply_x

def create_model():
  embed = C.layers.Embedding(emb_dim, name='embed')
  encoder = BiRecurrence(C.layers.GRU(hidden_dim//2), C.layers.GRU(hidden_dim//2))
  recoder = BiRecurrence(C.layers.GRU(hidden_dim//2), C.layers.GRU(hidden_dim//2))
  emit_enc = C.layers.Dense(num_labels, name='classify')
  emit_rec = C.layers.Dense(num_labels, name='classify')
  do_enc = C.layers.Dropout(0.5)
  do_rec = C.layers.Dropout(0.5)
  
  def recode(x, t, m):
    inp = embed(x)
    inp = C.layers.LayerNormalization()(inp)
    
    enc = encoder(inp)
    rec = enc
    
    em_enc = emit_enc(do_enc(enc))
    em_avg = emit_enc(t)
    
    lam = 0.5
    dec = C.ops.softmax(lam*em_enc + (1-lam)*em_avg)
    dec = m * dec
    return enc, rec, dec
  return recode

def create_criterion_function_preferred(model, labels):
  ce   = -C.reduce_sum(labels*C.ops.log(model))
  errs = C.classification_error(model, labels)
  return ce, errs

def enhance_data(data, enc, rec, is_readout=False):
  # First iteration, initialize masks
  clip = 1 if is_readout else 0
  masks = C.ops.clip(C.ops.argmax(y), clip, 1).eval({y: data[y]})
  for i in range(len(masks)):
    masks[i] = masks[i][..., None]
  s = C.io.MinibatchSourceFromData(dict(m=(masks, C.layers.typing.Sequence[C.layers.typing.tensor])))
  mems = s.next_minibatch(minibatch_size)
  data[m] = mems[s.streams['m']]
  guesses = enc.eval({x: data[x]})
  inputs = C.ops.argmax(x).eval({x: data[x]})
  tables = []
  for i in range(len(inputs)):
    ts = []
    table = {}
    counts = {}
    for j in range(len(inputs[i])):
      inp = int(inputs[i][j])
      if inp not in table:
        table[inp] = guesses[i][j]
        counts[inp] = 1
      else:
        table[inp] += guesses[i][j]
        counts[inp] += 1
    for inp in table:
      table[inp] /= counts[inp]
    for j in range(len(inputs[i])):
      inp = int(inputs[i][j])
      ts.append(table[inp])
    tables.append(np.array(np.float32(ts)))
  s = C.io.MinibatchSourceFromData(dict(t=(tables, C.layers.typing.Sequence[C.layers.typing.tensor])))
  mems = s.next_minibatch(minibatch_size)
  data[t] = mems[s.streams['t']]

def create_trainer():
  loss, label_error = create_criterion_function_preferred(dec, y)

  schedule_step = print_freq
  lr_per_sample = [2e-3]*2*schedule_step + [1e-3]*2*schedule_step + [5e-4]
  lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
  lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)

  momentum_as_time_constant = C.momentum_as_time_constant_schedule(1000)
  learner = C.adam(parameters=dec.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

  progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs)
  trainer = C.Trainer(dec, (loss, label_error), learner, progress_printer)
  if restore:
    trainer.restore_from_checkpoint("model-5.cntk")
  C.logging.log_number_of_parameters(dec)
  return trainer

def create_reader(path, is_training):
  return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
      source        = C.io.StreamDef(field='S0', shape=vocab_size, is_sparse=True), 
      slot_labels   = C.io.StreamDef(field='S1', shape=num_labels, is_sparse=True)
  )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

def validate():
  valid_reader = create_reader(files['valid']['file'], is_training=False)
  while True:
    data = valid_reader.next_minibatch(minibatch_size, input_map={
        x: valid_reader.streams.source,
        y: valid_reader.streams.slot_labels
    })
    if not data:
      break
    enhance_data(data, enc, rec)
    trainer.test_minibatch(data)
  trainer.summarize_test_progress()

def evaluate(epoch=0, write=False):
  test_reader = create_reader(files['test']['file'], is_training=False)
  if not write:
    while True:
      data = test_reader.next_minibatch(minibatch_size, input_map={
          x: test_reader.streams.source,
          y: test_reader.streams.slot_labels
      })
      if not data:
        break
      # Enhance data
      enhance_data(data, enc, rec)
      # Test model
      trainer.test_minibatch(data)
    trainer.summarize_test_progress()
    return
  # Else, do the above with file-writing (and lower minibatch size)
  with open("results-" + str(epoch) + ".csv", "w") as f:
    while True:
      data = test_reader.next_minibatch(minibatch_size//2, input_map={
          x: test_reader.streams.source,
          y: test_reader.streams.slot_labels
      })
      if not data:
        break
      # Enhance data
      enhance_data(data, enc, rec)
      # Test model
      trainer.test_minibatch(data)
      # Compute and write more detailed statistics
      logits = dec.eval({x: data[x], t: data[t], m: data[m]})
      sources = C.ops.argmax(x).eval({x: data[x]})
      labels  = C.ops.argmax(y).eval({y: data[y]})
      for i in range(len(sources)):
        seq_logits = logits[i]
        seq_sources = np.int32(sources[i])
        seq_labels = np.int32(labels[i])
        for j in range(len(seq_sources)):
          logit_ix = seq_logits[j].argmax()
          key = source_wl[seq_sources[j]]
          label = target_wl[seq_labels[j]]
          guess = target_wl[logit_ix]
          true_prob = seq_logits[j][seq_labels[j]]
          guess_prob = seq_logits[j][logit_ix]
          if np.isnan(true_prob):
            continue;
          true_rank = sorted(seq_logits[j], reverse=True).index(true_prob)
          f.write("\"%s\",%s,%s,%.4f,%.4f,%d\n" % (key, label, guess, true_prob, guess_prob, true_rank))
  trainer.summarize_test_progress()

def train():
  train_reader = create_reader(files['train']['file'], is_training=True)
  step = 0
  for epoch in range(num_epochs):
    epoch_end = (epoch+1) * epoch_size
    while step < epoch_end:
      data = train_reader.next_minibatch(minibatch_size, input_map={
          x: train_reader.streams.source,
          y: train_reader.streams.slot_labels
      })
      # Enhance data
      enhance_data(data, enc, rec)
      # Train model
      trainer.train_minibatch(data)
      step += data[y].num_samples
    trainer.summarize_training_progress()
    if (epoch + 1) % print_freq == 0:
      true_epoch = (epoch + 1)//print_freq
      if not restore:
        trainer.save_checkpoint("model-" + str(true_epoch) + ".cntk")
      validate()
      evaluate(true_epoch, true_epoch == 1 or true_epoch >= 5 and (true_epoch % 2 == 1 or true_epoch == 10))

def run_seq(seq):
  s = [source_dict[s] if s in source_dict else source_dict["_UNKNOWN_"] for s in seq.split()] # convert to word indices
  onehot = np.zeros([len(s),len(source_dict)], np.float32)
  for q in range(len(s)):
      onehot[q,s[q]] = 1

  out_dummy = np.float32(np.zeros((len(s), num_labels)))
  sIn = C.io.MinibatchSourceFromData(dict(xx=([onehot], C.layers.typing.Sequence[C.layers.typing.tensor]),
                                          yy=([out_dummy], C.layers.typing.Sequence[C.layers.typing.tensor])))
  mb = sIn.next_minibatch(len(onehot))
  data = {x: mb[sIn.streams['xx']], y: mb[sIn.streams['yy']]}
  enhance_data(data, enc, rec, True)

  best = C.ops.argmax(dec).eval({x: data[x], t: data[t], m: data[m]})[0]
  print()
  for b, tt in list(zip(seq.split(), [target_wl[int(b)] for b in best])):
    print("%s " % b, end='')
    if tt != "O" and bool(regex.match(b.strip())) and b.strip() not in keywords:
      tt = tt[1:len(tt)-1]
      print(": %s " % tt, end='')
  print()

model = create_model()
enc, rec, dec = model(x, t, m)
trainer = create_trainer()
train()

run_seq('private static _lpad ( value , columnWidth , fill = stringlit ) { let result = stringlit ; for ( let i = numlit ; i < columnWidth - value . length ; i ++ ) { result += fill ; } return result + value ; }')
