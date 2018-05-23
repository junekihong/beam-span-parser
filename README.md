# Linear-Time Span-Based Neural Constituency Parser

A DP beam-search extension of Mitchell Stern's span-based neural constituency parser, based on the [code from his github](https://github.com/mitchellstern/minimal-span-parser).

## Requirements and Setup

* Python 3.5 or higher.
* [DyNet](https://github.com/clab/dynet). We recommend installing DyNet from source with MKL support for significantly faster run time.

## Training

A new model can be trained using the command `python3 src/main.py train ...` with the following arguments:
Argument | Description | Default
--- | --- | ---
`--numpy-seed` | NumPy random seed | Random
`--parser-type` | `beam-parse` or `chart` | N/A
`--tag-embedding-dim` | Tag embedding dimension | 50
`--word-embedding-dim` | Word embedding dimension | 100
`--lstm-layers` | Number of bidirectional LSTM layers | 2
`--lstm-dim` | Hidden dimension of each LSTM within each layer | 250
`--label-hidden-dim` | Hidden dimension of label-scoring feedforward network | 250
`--dropout` | Dropout rate for LSTMs | 0.4
`--model-path-base` | Path base to use for saving models | N/A
`--train-path` | Path to training trees | `data/02-21.10way.clean`
`--dev-path` | Path to development trees | `data/22.auto.clean`
`--batch-size` | Number of examples per training update | 10
`--epochs` | Number of training epochs | No limit
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--print-vocabs` | Print the vocabularies before training | Do not print the vocabularies
`--beamsize` | Beam size | Infinity
`--cross-span` | Our new augmented loss | False
`--nocubepruning`* | Turns off cubepruning | False

\*This argument only applies to the beam parser.

Any of the DyNet command line options can also be specified.

The training and development trees are assumed to have predicted part-of-speech tags.

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

To train our parser with beamsize 20 using the default hyperparameters, you can use the command:

```
python3 src/main.py train --parser-type beam-parse --model-path-base models/beam-parse-model --epochs 50 --beamsize 20 --cross-span
```

Alternatively, to train a chart parser (replicating stern et al. 2017), you can use the command:

```
python3 src/main.py train --parser-type chart --model-path-base models/chart-model
```

## Evaluation

A saved model can be evaluated on a test corpus using the command `python3 src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--test-path` | Path to test trees | `data/23.auto.clean`
`--beamsize` | Beam size (for separate decoding) | The trained model's beam size
`--nocubepruning`* | Turns off cubepruning | False

As above, any of the DyNet command line options can also be specified.

The test trees are assumed to have predicted part-of-speech tags.

As an example, after extracting the pre-trained top-down model, you can evaluate it on the test set using the following command:

```
python3 src/main.py test --model-path-base models/action-b020-model_dev=92.20
```

