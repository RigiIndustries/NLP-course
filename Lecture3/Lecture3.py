from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

base = Path(__file__).parent
docs = [
    (base / "IMDB_1.txt").read_text(encoding="utf-8"),
    (base / "IMDB_2.txt").read_text(encoding="utf-8"),
    (base / "IMDB_3.txt").read_text(encoding="utf-8"),
]
labels = ["Doc1", "Doc2", "Doc3"]

# bag of words / stop words removed
cv = CountVectorizer(stop_words="english")
X = cv.fit_transform(docs)
vocab = list(cv.get_feature_names_out())
bow = X.toarray()

print("\n--- Bag of Words (stop words removed) ---")
print(f"Vocabulary size: {len(vocab)}")
print("Vocabulary:", ", ".join(vocab))

print("\nCounts per doc:")
for i, row in enumerate(bow):
    counts = {term: int(row[j]) for j, term in enumerate(vocab) if row[j] > 0}
    print(f"{labels[i]}:", counts)

# --- 2) TF-IDF for 'rick' and 'artifacts' ---
tfidf = TfidfVectorizer(stop_words="english")
T = tfidf.fit_transform(docs)           # shape: 3 x |V|
terms = {t: j for j, t in enumerate(tfidf.get_feature_names_out())}

def tfidf_scores(term: str):
    t = term.lower()
    if t not in terms:
        return [0.0, 0.0, 0.0]
    col = T[:, terms[t]].toarray().ravel()
    return [round(float(x), 4) for x in col]

print("\n--- TF-IDF ---")
for term in ["rick", "artifacts"]:
    print(f"{term}: {dict(zip(labels, tfidf_scores(term)))}")

print("\nNotes:")
print("- 'rick' appears in Doc1 and Doc3, so Doc2 is 0.0")
print("- 'artifacts' appears only in Doc3, so Doc1 and Doc2 are 0.0")
