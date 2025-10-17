from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

base = Path(__file__).parent
files = [base/"IMDB_1.txt", base/"IMDB_2.txt", base/"IMDB_3.txt"]
labels = ["Doc1", "Doc2", "Doc3"]
docs = [p.read_text(encoding="utf-8") for p in files]

# BIGRAMS only, stop words removed; scikit-learn handles punctuation/spacing
vec = TfidfVectorizer(ngram_range=(2,2), stop_words="english")
X = vec.fit_transform(docs)   # shape: 3 x |V_bigrams|
terms = vec.get_feature_names_out()

# show top-k bigrams per doc
k = 8
print("\n=== Top TF-IDF bigrams per document ===")
for i, label in enumerate(labels):
    row = X[i].toarray().ravel()
    idx = np.argsort(row)[::-1]
    tops = [(terms[j], round(row[j], 4)) for j in idx if row[j] > 0][:k]
    print(f"{label}: {tops}")

# If you need specific bigram scores, list them here:
want = ["construction site", "rick grimes", "search artifacts", "join forces"]
term_index = {t:i for i,t in enumerate(terms)}
print("\n=== Selected bigram scores (if present) ===")
for phrase in want:
    j = term_index.get(phrase)
    if j is None:
        print(f"{phrase!r}: not in vocabulary")
    else:
        scores = X[:, j].toarray().ravel()
        print(f"{phrase!r}: {{Doc1: {scores[0]:.4f}, Doc2: {scores[1]:.4f}, Doc3: {scores[2]::.4f}}}")
