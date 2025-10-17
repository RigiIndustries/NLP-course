from collections import Counter
import re
import nltk

def _ensure(nltk_id, path_hint):
    try:
        nltk.data.find(path_hint)
    except LookupError:
        nltk.download(nltk_id)

_ensure("gutenberg", "corpora/gutenberg")
_ensure("punkt", "tokenizers/punkt")
_ensure("punkt_tab", "tokenizers/punkt_tab")
_ensure("stopwords", "corpora/stopwords")

from nltk.corpus import gutenberg, stopwords
from nltk import word_tokenize
from nltk.util import ngrams

# --- load Macbeth ---
text = gutenberg.raw("shakespeare-macbeth.txt")

# --- clean ---
stops = set(stopwords.words("english"))
tokens = [t.lower() for t in word_tokenize(text) if re.fullmatch(r"\w+", t)]
tokens = [t for t in tokens if t not in stops]

# --- build n-grams ---
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# --- top frequencies ---
top_k = 25
bi_counts = Counter(bigrams).most_common(top_k)
tri_counts = Counter(trigrams).most_common(top_k)

print("\n=== Top bigrams (after cleaning) ===")
for (w1, w2), c in bi_counts:
    print(f"{w1} {w2}: {c}")

print("\n=== Top trigrams (after cleaning) ===")
for (w1, w2, w3), c in tri_counts:
    print(f"{w1} {w2} {w3}: {c}")

# Optional: save CSVs
try:
    import csv
    with open("macbeth_bigrams.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["bigram","count"])
        w.writerows([[" ".join(bg), c] for bg, c in bi_counts])
    with open("macbeth_trigrams.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["trigram","count"])
        w.writerows([[" ".join(tg), c] for tg, c in tri_counts])
    print("\nSaved: macbeth_bigrams.csv, macbeth_trigrams.csv")
except Exception:
    pass
