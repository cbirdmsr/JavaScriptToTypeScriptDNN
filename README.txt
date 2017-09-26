All code files are in a Code subdirectory.

Step 1: Run crawler.py to get the top 1K repos and download them into a directory named Clones. Provided the original repos.csv to clone those projects instead, though some may have been removed and they won't be at the same state.

Step 2: Run rnn-prep.js to generate one file per project in a directory named outputs-project. If wanting to replicate the CheckJS experiments, first make a copy of the directory 'Clones' named 'Clones-cleaned'. Run checkjs-pre.js and checkjs-post.js if also interested in replicating the CheckJS experiments.

Step 3: Run lexer.py (alter vocab cut-offs and input file at the top if needed). It will generate vocabularies in source.wl and target.wl and train/valid/test files in seq2seq-*.txt. Follow running lexer by:
	py txt2ctf.py --map source.wl target.wl --input seq2seq-train.txt --output seq2seq-train.ctf
	py txt2ctf.py --map source.wl target.wl --input seq2seq-valid.txt --output seq2seq-valid.ctf
	py txt2ctf.py --map source.wl target.wl --input seq2seq-test.txt --output seq2seq-test.ctf
to convert to CNTK input format (txt2ctf is from CNTK). It also write 'test_projects.txt' with the list of test_projects for comparison with CheckJS.

Step 4: Run infer.py to train and test model. It will generate model files and CSVs with test results at epochs 1, 5, 7, 9 and 10. Model files with the plain RNN and DTs pre-trained models are in Models/. Their results are in Results/results-*.csv and a file to copy those into to get detailed statistics is in there as well (TypeRankings.csv). Finally, Charts.csv contains some experimental results including the Zipfian distribution, a place to past into the output of evaluation.py (replace spaces with tabs) to get comparison with CheckJS and a place to generate precision/recall curves.

After it has been trained, choose any of the resulting CSV files and plug its name into files consistency.py and (if comparing with CheckJS) 'evaluation.py' at the top to generate those results also. tokentypemodel.py contains code for the naive baseline.

Step 5: readout.py contains code that is also in the LSTM directory of the web-page and responds to sequences by running the RNN (DT) and writing to file the results. refiner.py contains some old code to iterate with CheckJS and may be perused without guarantees.