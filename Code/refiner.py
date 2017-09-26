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

if len(sys.argv) < 2:
  print("Not enough arguments, pass file name")
  exit(1)
inp = sys.argv[1]
outp = inp[:len(inp) - inp[::-1].index(".")] + "csv"
source_file = inp[:len(inp) - inp[::-1].index("\\")] + "source.wl"
target_file = inp[:len(inp) - inp[::-1].index("\\")] + "target.wl"
model_file = inp[:len(inp) - inp[::-1].index("\\")] + "model-1-10.cntk"

# load dictionaries
source_wl = [line.rstrip('\n') for line in open(source_file)]
target_wl = [line.rstrip('\n') for line in open(target_file)]
source_dict = {source_wl[i]:i for i in range(len(source_wl))}
target_dict = {target_wl[i]:i for i in range(len(target_wl))}

# number of words in vocab, slot labels, and intent labels
vocab_size = len(source_dict)
num_labels = len(target_dict)
print_freq = 100 # Number of steps per epoch, 
epoch_size = 3043719//print_freq # Total #tokens, tentatively //print_freq for increased print frequency
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
  do_enc = C.layers.Dropout(0.0)
  do_rec = C.layers.Dropout(0.0)
  
  def recode(x, t, m):
    inp = embed(x)
    inp = C.layers.LayerNormalization()(inp)
    
    enc = encoder(inp)
    em_enc = emit_enc(do_enc(enc))
    em_avg = emit_enc(t)
    
    rec = enc
    
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
    w = "_UNKNOWN_"
    if value.strip() in source_dict:
      w = value.strip()
    elif ttype in Comment:
      continue
    elif ttype in Literal:
      if ttype in Literal.String:
        w = "stringlit"
      elif ttype in Literal.Number:
        w = "numlit"
    ws.append(w)
  return txt, ws

def run_checkJS(txt):
  sep = chr(31)
  with open('temp.js', 'w') as f:
    f.write(txt)
  success = execute_js('checkjs.js', 'temp.js')
  if not success:
    print("Error running checkJS!")
  with open('temp-out.txt', 'r') as f:
    types = [line.rstrip('\n').split(sep) for line in f]
  types = [target_dict[tt[1]] if tt[1] in target_dict else target_dict["$UNKNOWN$"] for tt in types]
  # Add start/end of sentence types
  types.insert(0, 0)
  types.append(0)
  return types

def train_with_checkJS(types, data):
  enhance_data(data, enc, rec, True)
  N = len(types)
  train_mask = np.zeros((N, 1), np.float32)
  for j in range(len(types)):
    if types[j] != 0 and types[j] not in exclude:
      train_mask[j][0] = 1
  outputs = scipy.sparse.csr_matrix((np.ones(N, np.float32), (range(N), types)), shape=(N, num_labels))
  # Set mask for training
  s = C.io.MinibatchSourceFromData(dict(y=(outputs, C.layers.typing.Sequence[C.layers.typing.tensor]),
                                        m=(train_mask, C.layers.typing.Sequence[C.layers.typing.tensor])))
  mems = s.next_minibatch(N)
  data[y] = mems[s.streams['y']]
  data[m] = mems[s.streams['m']]
  trainer.train_minibatch(data)

def write_preds(tokens, pred):
  sep = chr(31)
  with open(outp, 'w') as f:
    ix = 0
    for tt, v, in tokens:
      f.write("%s%s%s" % (v.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r"), sep, str(tt)[6:]))
      print(v, end='')
      if v.strip() == '' or t in Comment:
        f.write('\n')
        continue
      pr = pred[ix]
      ix += 1
      if v.strip() in keywords or not bool(regex.match(v.strip())):
        f.write('\n')
        continue
      r = [i[0] for i in sorted(enumerate(pr), key=lambda x:-x[1])]
      guess = target_wl[r[0]]
      gs = [target_wl[r[ix]] for ix in range(min(len(r),5))]
      gs = [g[1:len(g)-1] if g[0]=="$" else g for g in gs]
      if target_wl[r[0]] != "O" and target_wl[r[0]] != "$UNKNOWN$":
        print(" : %s" % guess[1:len(guess)-1], end='')
      f.write(",%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f,%s,%.4f\n" %\
            (gs[0], pr[r[0]], gs[1], pr[r[1]], gs[2], pr[r[2]], gs[3], pr[r[3]], gs[4], pr[r[4]]))
  print()

def iterate(tokens, types, predictions):
  if len(types) != len(predictions):
    return tokens, types, predictions
  def add(index, tt):
    tokens.insert(index + 1, tt)
    tokens.insert(index + 1, ":")
    predictions.insert(index + 1, ("O", 0.0))
    predictions.insert(index + 1, ("O", 0.0))
    types.insert(index + 1, "O")
    types.insert(index + 1, "O")
  inFunc = False
  inParams = False
  funcConfBuffer = 0.0
  funcTypeBuffer = ""
  for ix, (pred, conf) in enumerate(predictions):
    word = tokens[ix]
    # Simple heuristic to determine variable locations
    if word == "function" or word == "<s>" or (word in ["public", "private", "static"] and tokens[ix + 1] == "("):
      inFunc = True
    elif word == "(" and inFunc:
      inParams = True
    elif word == ")" and inFunc:
      inFunc = False
      inParams = False
      if funcTypeBuffer != "" and tokens[ix+1] != ":":
        add(ix, funcTypeBuffer)
        funcTypeBuffer = ""
    if types[ix] == "O":
      continue
    if conf > 0.99 and (ix + 1 == len(tokens) or tokens[ix + 1] != ":"):
      ttype = target_wl[int(pred)]
      ttype = ttype[1:len(ttype)-1]
      isFuncDecl = inFunc and not inParams
      isParam = inParams
      isVarDecl = tokens[ix - 1] in ["var", "let"]
      if isFuncDecl:
        funcTypeBuffer = ttype
        funcConfBuffer = conf
      elif isParam or isVarDecl:
        add(ix, ttype)
  return tokens, types, predictions

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
  
  pred = run_checkJS(txt)
  if N != len(pred):
    return []
  #train_with_checkJS(pred, data)
  pred = [(p, 0.0) for p in pred]
  pred_js = pred
  enhance_data(data, enc, rec, True)
  pred = C.ops.argmax(dec).eval({x: data[x], t: data[t], m: data[m]})[0]
  confs = C.ops.reduce_max(dec).eval({x: data[x], t: data[t], m: data[m]})[0]
  pred = [(pred[j], confs[j]) for j in range(len(pred))]
  #for i in range(len(pred_js)):
  #  if pred_js[i][0] not in exclude:
  #    pred[i] = pred_js[i]
  #write_preds(tokens, pred)
  return pred

model = create_model()
enc, rec, dec = model(x, t, m)
overall_hits = 0
overall_count = 0
fps = 0
index = 0
num_steps = 1
exclude = [target_dict["O"], target_dict["$any$"], target_dict["$any[]$"], target_dict["$any[][]$"]]
with open('seq2seq-test.txt', 'r') as f:
  for line in f:
    trainer = create_trainer()
    index += 1
    if index > 100:
      break
    parts = line.split('\t')
    tokens = parts[0]
    tokens = tokens.split(" ")
    types = parts[1]
    types = types.split(" ")
    predictions_init = run_seq(tokens[1:len(tokens)-1], 0)
    predictions = predictions_init
    for step in range(num_steps):
      tokens, types, predictions = iterate(tokens, types, predictions_init)
      predictions = run_seq(tokens[1:len(tokens)-1], step+1)
    if len(types) != len(predictions):
      if len(types) == len(predictions_init):
        predictions = predictions_init
      else:
        print("ERROR")
        continue
    count = 0
    hits = 0
    for i in range(len(types)):
      if types[i] == "O":
        continue
      count += 1
      pred, conf = predictions[i]
      if types[i] == target_wl[int(pred)]:
        hits += 1
      elif target_wl[int(pred)] != "O":
        fps += 1
    overall_hits += hits
    overall_count += count
    print("%d, %d: %.3f%%, %.3f%%" % (index, overall_count, 100*overall_hits/overall_count, 100*overall_hits/(overall_hits+fps)))
#with open(inp, 'r') as f:
#  content = f.read()
#run_seq(content)

