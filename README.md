# Hate speech detection with neural networks.

## Requirements

Before running 'run.py', you will need:
  - numpy
  - pandas
  - keras
  - sklearn
  - gensim
  - kerastuner

Then, you will also need data to train with and pre-trained embeddings.

## Example

The command below will run the model "LSTM" for 25 trials.\

```bash
python run.py -m lstm -t 25
```

Using the -l parameter will make use of the features' lexicon selected.\

```bash
python run.py -l sel -t 50
```

Using 'run_cv.py' instead of 'run.py' will perform Cross Validation.\
Use 'run.py -h' or 'run_cv.py -h' to show help about the input options.\
