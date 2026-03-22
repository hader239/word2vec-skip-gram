# Word2Vec Skip-Gram with Negative Sampling

From-scratch Word2Vec skip-gram in NumPy. Trains on War and Peace by Lev Tolstoy and evaluates via cosine similarity.

## How It Works

1. **Preprocessing** -- Lowercases text, strips non-alphabetic characters (Latin + Cyrillic), filters words below `MIN_COUNT`, and probabilistically discards frequent words (`subsample_t`).
2. **Pair generation** -- Creates (center, context) skip-gram pairs with a random window sampled from `[1, WINDOW_SIZE]` per word. Pairs are regenerated each epoch for training diversity.
3. **Training** -- Mini-batch SGD with frequency-based negative sampling (unigram^0.75 via a 10M-entry precomputed table), linear LR decay from `LEARNING_RATE` to `MIN_LR`.
4. **Output** -- Final embeddings are `(W_in + W_out) / 2`, saved to `embeddings.txt` in standard Word2Vec text format (loadable by gensim, etc.). Nearest neighbors printed via cosine similarity.

## Requirements

- Python 3.8+
- NumPy

```
pip install numpy
```

## Usage

```
python word2vec.py
```

### Configuration

Hyperparameters in `word2vec.py`:


| Parameter       | Default           | Description                            |
| --------------- | ----------------- | -------------------------------------- |
| `CORPUS_PATH`   | `warandpeace.txt` | Input text file                        |
| `EMBEDDING_DIM` | `100`             | Word vector dimensionality             |
| `WINDOW_SIZE`   | `5`               | Max context window radius              |
| `MIN_COUNT`     | `5`               | Min word frequency for vocabulary      |
| `NUM_NEGATIVES` | `10`              | Negative samples per positive pair     |
| `LEARNING_RATE` | `0.035`           | Initial SGD learning rate              |
| `MIN_LR`        | `0.0005`          | Final learning rate after linear decay |
| `BATCH_SIZE`    | `512`             | Mini-batch size                        |
| `EPOCHS`        | `50`              | Training epochs                        |
| `subsample_t`   | `1e-3`            | Frequent-word subsampling threshold    |


### Results

Trained on War and Peace (~190K tokens after subsampling, 6,347 vocab words), 50 epochs in ~640s:

```
князь:    андрей 0.92, василий 0.63, багратион 0.41
наташа:   соня 0.71, она 0.56, сказала 0.50
война:    союз 0.43, кампания 0.43, кончилась 0.42
солдат:   толпами 0.55, солдата 0.51, офицеров 0.50
любовь:   ближнему 0.48, счастье 0.44, храма 0.41
```

The model learns meaningful semantic relationships: "князь" (prince) maps to character names Andrei, Vasily, Bagration; "наташа" (Natasha) maps to her friend Sonya; "солдат" (soldier) clusters with military terms.