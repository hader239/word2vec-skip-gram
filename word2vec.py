import numpy as np
import re
import time
from collections import Counter


# ─── Preprocessing ───────────────────────────────────────────────────────────

def load_and_preprocess(filepath, min_count=2, subsample_t=1e-3):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().lower()

    text = re.sub(r"[^a-zа-яё ]", " ", text)
    tokens = text.split()

    counts = Counter(tokens)
    vocab = sorted(w for w, c in counts.items() if c >= min_count)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    total = sum(counts[w] for w in vocab)
    freq = {w: counts[w] / total for w in vocab}

    filtered = []
    for t in tokens:
        if t not in word2idx:
            continue
        f = freq[t]
        keep_prob = np.sqrt(subsample_t / f) + subsample_t / f
        if np.random.random() < keep_prob:
            filtered.append(word2idx[t])

    corpus = np.array(filtered, dtype=np.int32)
    return corpus, word2idx, idx2word, counts


# ─── Training Data ───────────────────────────────────────────────────────────

def generate_pairs(corpus, window_size):
    n = len(corpus)
    windows = np.random.randint(1, window_size + 1, size=n)
    pairs = []
    for d in range(1, window_size + 1):
        mask = windows[: n - d] >= d
        centers = corpus[: n - d][mask]
        contexts = corpus[d:][mask]
        pairs.append(np.stack([centers, contexts], axis=1))
        mask = windows[d:] >= d
        centers = corpus[d:][mask]
        contexts = corpus[: n - d][mask]
        pairs.append(np.stack([centers, contexts], axis=1))
    return np.concatenate(pairs)


# ─── Model ────────────────────────────────────────────────────────────────────

def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    exp_neg = np.exp(-x[pos])
    out[pos] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(x[~pos])
    out[~pos] = exp_pos / (1.0 + exp_pos)
    return out


def build_noise_distribution(counts, word2idx, power=0.75):
    vocab_size = len(word2idx)
    dist = np.zeros(vocab_size)
    for w, idx in word2idx.items():
        dist[idx] = counts[w] ** power
    dist /= dist.sum()
    return dist


def train(corpus, window_size, vocab_size, noise_dist, embedding_dim=100,
          num_negatives=5, lr=0.025, min_lr=0.0005, batch_size=512, epochs=1):
    scale = 0.5 / embedding_dim
    W_in = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))
    W_out = np.zeros((vocab_size, embedding_dim))

    noise_table = np.random.choice(
        vocab_size, size=10_000_000, replace=True, p=noise_dist
    )
    table_len = len(noise_table)

    pilot_pairs = generate_pairs(corpus, window_size)
    n_pairs = len(pilot_pairs)
    del pilot_pairs
    total_batches = (n_pairs + batch_size - 1) // batch_size
    log_every = max(1, total_batches // 20)
    global_steps = epochs * total_batches

    for epoch in range(1, epochs + 1):
        pairs = generate_pairs(corpus, window_size)
        n_pairs = len(pairs)
        total_batches = (n_pairs + batch_size - 1) // batch_size
        np.random.shuffle(pairs)
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, start in enumerate(range(0, n_pairs, batch_size), 1):
            global_step = (epoch - 1) * total_batches + batch_idx
            current_lr = lr - (lr - min_lr) * (global_step / global_steps)

            batch = pairs[start : start + batch_size]
            B = len(batch)
            centers = batch[:, 0]
            contexts = batch[:, 1]

            offsets = np.random.randint(0, table_len - num_negatives, size=B)
            negatives = np.array([noise_table[o:o + num_negatives] for o in offsets])

            # Forward
            h = W_in[centers]                               # (B, D)
            v_ctx = W_out[contexts]                         # (B, D)
            v_neg = W_out[negatives]                        # (B, K, D)

            pos_score = np.sum(h * v_ctx, axis=1)           # (B,)
            neg_score = np.einsum("bd,bkd->bk", h, v_neg)  # (B, K)

            sig_pos = sigmoid(pos_score)
            sig_neg = sigmoid(neg_score)

            loss = -np.sum(np.log(sig_pos + 1e-10)) - np.sum(
                np.log(1.0 - sig_neg + 1e-10)
            )
            epoch_loss += loss
            epoch_samples += B

            # Backward
            g_pos = sig_pos - 1.0                            # (B,)
            g_neg = sig_neg                                  # (B, K)

            grad_h = g_pos[:, None] * v_ctx + np.einsum("bk,bkd->bd", g_neg, v_neg)
            grad_ctx = g_pos[:, None] * h                    # (B, D)
            grad_neg = g_neg[:, :, None] * h[:, None, :]     # (B, K, D)

            np.add.at(W_in, centers, -current_lr * grad_h)
            np.add.at(W_out, contexts, -current_lr * grad_ctx)
            np.add.at(
                W_out,
                negatives.ravel(),
                -current_lr * grad_neg.reshape(-1, embedding_dim),
            )

            if batch_idx % log_every == 0:
                batch_avg = loss / B
                pct = 100.0 * batch_idx / total_batches
                print(f"  Epoch {epoch}/{epochs} [{pct:5.1f}%] Loss: {batch_avg:.4f}  lr: {current_lr:.6f}")

        print(f"  Epoch {epoch}/{epochs} complete | Avg Loss: {epoch_loss / n_pairs:.4f}")

    return W_in, W_out


# ─── Evaluation ───────────────────────────────────────────────────────────────

def cosine_similarities(embeddings, vec):
    norms = np.linalg.norm(embeddings, axis=1)
    norms = np.maximum(norms, 1e-10)
    vec_norm = max(np.linalg.norm(vec), 1e-10)
    return (embeddings @ vec) / (norms * vec_norm)


def most_similar(word, word2idx, idx2word, embeddings, top_n=10):
    if word not in word2idx:
        print(f"  '{word}' not in vocabulary")
        return []
    vec = embeddings[word2idx[word]]
    sims = cosine_similarities(embeddings, vec)
    top = np.argsort(sims)[::-1][1 : top_n + 1]
    return [(idx2word[i], float(sims[i])) for i in top]


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(1) 

    CORPUS_PATH = "warandpeace.txt"
    EMBEDDING_DIM = 100
    WINDOW_SIZE = 5
    MIN_COUNT = 5
    NUM_NEGATIVES = 10
    LEARNING_RATE = 0.035
    BATCH_SIZE = 512
    EPOCHS = 50

    print("Loading and preprocessing...")
    corpus, word2idx, idx2word, counts = load_and_preprocess(CORPUS_PATH, MIN_COUNT)
    vocab_size = len(word2idx)
    print(f"  Vocab: {vocab_size} words | Corpus: {len(corpus)} tokens")

    print("Building noise distribution...")
    noise_dist = build_noise_distribution(counts, word2idx)

    print(f"Training ({EPOCHS} epochs, fresh pairs each epoch)...")
    t0 = time.time()
    W_in, W_out = train(
        corpus,
        WINDOW_SIZE,
        vocab_size,
        noise_dist,
        embedding_dim=EMBEDDING_DIM,
        num_negatives=NUM_NEGATIVES,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    embeddings = (W_in + W_out) / 2

    out_path = "embeddings.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{vocab_size} {EMBEDDING_DIM}\n")
        for i in range(vocab_size):
            vec_str = " ".join(f"{v:.6f}" for v in embeddings[i])
            f.write(f"{idx2word[i]} {vec_str}\n")
    print(f"  Saved to {out_path}")

    print("\n=== Most Similar Words ===")
    for query in ["князь", "война", "наташа", "солдат", "любовь"]:
        results = most_similar(query, word2idx, idx2word, embeddings)
        if results:
            print(f"\n  {query}:")
            for w, s in results[:5]:
                print(f"    {w:20s} {s:.4f}")
