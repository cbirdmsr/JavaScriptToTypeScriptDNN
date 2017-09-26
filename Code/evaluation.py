from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import requests
import os
import sys

import math
import numpy as np
import scipy.sparse
import cntk as C
import pygments

import re
import string
from pygments.lexers import TypeScriptLexer
from pygments.token import Comment, Literal

from Naked.toolshed.shell import execute_js, muterun_js

regex = re.compile(r"^[^\d\W]\w*$", re.UNICODE)
keywords = ["async", "await", "break", "continue", "class", "extends", "constructor", "super", "extends", "const", "let", "var", "debugger", "delete", "do", "while", "export", "import", "for", "each", "in", "of", "function", "return", "get", "set", "if", "else", "instanceof", "typeof", "null", "undefined", "switch", "case", "default", "this", "true", "false", "try", "catch", "finally", "void", "yield", "any", "boolean", "null", "never", "number", "string", "symbol", "undefined", "void", "as", "is", "enum", "type", "interface", "abstract", "implements", "static", "readonly", "private", "protected", "public", "declare", "module", "namespace", "require", "from", "of", "package"]
exclude = ["O", "$any$", "$any[]$", "$any[][]$"]

source_file = "source.wl"
target_file = "target.wl"
model_file = "model-5.cntk"

# load dictionaries
source_wl = [line.rstrip('\n') for line in open(source_file)]
target_wl = [line.rstrip('\n') for line in open(target_file)]
source_dict = {source_wl[i]:i for i in range(len(source_wl))}
target_dict = {target_wl[i]:i for i in range(len(target_wl))}

# number of words in vocab, slot labels, and intent labels
vocab_size = len(source_dict)
num_labels = len(target_dict)
print_freq = 10 # Number of steps per epoch, 
epoch_size = 4291262//print_freq # Total #tokens, tentatively //print_freq for increased print frequency
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

  schedule_step = 1 * print_freq
  lr_per_sample = [1e-3]
  lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
  lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)

  momentum_as_time_constant = C.momentum_as_time_constant_schedule(0)
  learner = C.adam(parameters=dec.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

  trainer = C.Trainer(dec, (loss, label_error), learner)
  trainer.restore_from_checkpoint(model_file)
  return trainer

def prep(tokens):
  ws = []
  txt = ''
  for ttype, value in tokens:
    txt += value
    if value.strip() == '':
      continue
    # TypeScript lexer fails on arrow token
    if len(ws) > 0 and ws[len(ws)-1] == "=" and value == ">":
      ws[len(ws)-1] = "=>"
      continue
    elif len(ws) > 1 and ws[len(ws)-2] == "." and ws[len(ws)-1] == "." and value == ".":
      ws[len(ws)-2] = "..."
      ws.pop()
      continue
    elif len(ws) > 1 and ws[len(ws)-2] == "`" and value == "`":
      ws[len(ws)-2] = "`" + ws[len(ws)-1] + "`"
      ws.pop()
      continue
    w = "_UNKNOWN_"
    if value.strip() in source_dict:
      w = value.strip()
    elif ttype in Comment:
      continue
    elif ttype in Literal:
      if ttype in Literal.String:
        if value != '`':
          w = "\"s\""
        else:
          w = '`'
      elif ttype in Literal.Number:
        w = "0"
    ws.append(w)
  return txt, ws

def run_seq(seq, pass_no):
  # Set up data
  tokens = list(pygments.lex(" ".join(seq), TypeScriptLexer()))
  txt, ws = prep(tokens)
  ws.insert(0, "<s>")
  ws.append("</s>")
  # Set up tensors
  inputs = np.zeros(len(ws))
  outputs = np.zeros(len(ws))
  for i in range(len(ws)):
    inputs[i] = source_dict[ws[i]]
  N = len(inputs)
  inputs = scipy.sparse.csr_matrix((np.ones(N, np.float32), (range(N), inputs)), shape=(N, vocab_size))
  outputs = scipy.sparse.csr_matrix((np.ones(N, np.float32), (range(N), outputs)), shape=(N, num_labels))
  sIn = C.io.MinibatchSourceFromData(dict(xx=([inputs], C.layers.typing.Sequence[C.layers.typing.tensor]),
                                          yy=([outputs], C.layers.typing.Sequence[C.layers.typing.tensor])))
  mb = sIn.next_minibatch(N)
  data = {x: mb[sIn.streams['xx']], y: mb[sIn.streams['yy']]}
  
  enhance_data(data, enc, rec, True)
  pred = C.ops.argmax(dec).eval({x: data[x], t: data[t], m: data[m]})[0]
  confs = C.ops.reduce_max(dec).eval({x: data[x], t: data[t], m: data[m]})[0]
  pred = [(pred[j], confs[j]) for j in range(len(pred))]
  return pred

model = create_model()
enc, rec, dec = model(x, t, m)
index = 0
num_steps = 0
with open('test_projects.txt', 'r') as f:
  test_projects = [line.rstrip() for line in f]

for project in test_projects:
  hits_cj = 0
  hits_dl = 0
  hits_mix = 0
  tps = 0
  fps = 0
  count = 0
  true_file = "outputs-checkjs-pre/" + project
  checkJS_file = "outputs-checkjs-post/" + project
  trainer = create_trainer()
  checkJS_types = {}
  with open(checkJS_file, 'r') as f:
    for l in f:
      split = l.rstrip().split("\t")
      if len(split) < 2:
        continue
      tokens = split[0]
      types = split[1].split(" ")
      checkJS_types[tokens] = types
  print(project, end=' ')
  with open(true_file, 'r') as f:
    for l in f:
      index += 1
      split = l.rstrip().split("\t")
      if len(split) < 2:
        print("S", end='')
        continue
      tokens = split[0]
      types = split[1].split(" ")
      # Get types from CheckJS
      if tokens not in checkJS_types:
        print("N", end='')
        continue
      cj_types = checkJS_types[tokens]
      if len(types) != len(cj_types):
        print("D", end='')
        continue
      # Set up tokens for DL
      tokens = tokens.split(" ")
      tokens.insert(0, "<s>")
      tokens.append("</s>")
      types.insert(0, "O")
      types.append("O")
      cj_types.insert(0, "O")
      cj_types.append("O")
      for i in range(len(types)):
        if types[i] not in target_dict:
          types[i] = "$UNKNOWN$"
        if cj_types[i] not in target_dict:
          cj_types[i] = "$UNKNOWN$"
      # Run deep learner
      try:
        predictions = run_seq(tokens[1:len(tokens)-1], 0)
      except ValueError:
        print("E", end='')
        continue
      predictions_init = predictions
      for step in range(num_steps):
        tokens, types, predictions = iterate(tokens, types, predictions_init)
        predictions = run_seq(tokens[1:len(tokens)-1], step+1)
      if len(types) != len(predictions):
        if len(types) == len(predictions_init):
          predictions = predictions_init
        else:
          print("E", end='')
          break
      # Get stats
      for i in range(len(types)):
        if types[i] == "O":
          continue
        pred, conf = predictions[i]
        choice = cj_types[i]
        # Choose hybrid type
        overruled = False
        if conf > 0.9:# and cj_types[i] in exclude:
          choice = target_wl[int(pred)]
          overruled = True
        count += 1
        if types[i] == cj_types[i]:
          hits_cj += 1
        if types[i] == target_wl[int(pred)]:
          hits_dl += 1
        if types[i] == choice:
          hits_mix += 1
          if types[i] != cj_types[i]:
            tps += 1
        elif types[i] == cj_types[i]:
          fps += 1
    
    pct_cj = 0 if count == 0 else 100*hits_cj/count
    pct_dl = 0 if count == 0 else 100*hits_dl/count
    pct_mix = 0 if count == 0 else 100*hits_mix/count
    pct_fps = 0 if count == 0 else 100*fps/count
    pct_tps = 0 if count == 0 else 100*tps/count
    print(" %d %d %.3f%% %.3f%% %.3f%% %.3f%% %.3f%%" % (index, count, pct_cj, pct_dl, pct_mix, pct_tps, pct_fps))

