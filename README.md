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

```bash
python run.py -m bilstm_cnn -t 25
```

This command will run the model 'BiLSTM + CNN' for 25 trials.\
Using 'run_cv_2.py' instead of 'run.py' will perform Cross Validation.\
Using 'run_partitions.py' will perform CV in an already splitted dataset.\
'run_LIWC.py' and 'run_SEL.py' use lexicon features as input, but don't perform CV nor hyperparameter optimization.\

## WARNING!

Do not use 'run_cv.py' since it performs CV in a wrong way. Use 'run_cv_2.py' instead.

