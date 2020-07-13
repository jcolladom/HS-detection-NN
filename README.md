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

This command will run the model 'BiLSTM + CNN' for 25 trials.
