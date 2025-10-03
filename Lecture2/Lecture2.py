import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize

import nltk

# ---------------------- Making sure everything runs ---------------------------
def _ensure(nltk_id, path_hint):
    try:
        nltk.data.find(path_hint)
    except LookupError:
        nltk.download(nltk_id)

_ensure("punkt", "tokenizers/punkt")
_ensure("punkt_tab", "tokenizers/punkt_tab")
_ensure("stopwords", "corpora/stopwords")
_ensure("wordnet", "corpora/wordnet")
_ensure("omw-1.4", "corpora/omw-1.4")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng")
        except:
            nltk.download("averaged_perceptron_tagger")

# ------------------------------------------------------------------------------

text = "When Alexander Graham Bell invented the telephone, he had three missed calls from Chuck Norris."

# 1) Tokenization
# (a) whitespace
tokens_ws = text.split()
print("Whitespace tokens:", tokens_ws)

# (b) Regex, only words that start with a capital letter
tokens_caps = re.findall(r"\b[A-Z][a-zA-Z\-']*\b", text)
print("Capitalized tokens:", tokens_caps)

# 2) remove Stopwords
stops = set(stopwords.words("english"))
tokens = [t for t in word_tokenize(text) if re.match(r"\w+", t)]
tokens_no_stop = [t for t in tokens if t.lower() not in stops]
print("No stop tokens:", tokens_no_stop)

# 3) Stemming and Lemmatization
stemmer = PorterStemmer()
stems = [stemmer.stem(t) for t in tokens_no_stop]
print("Stems:", stems)

# Better lemmatization if we map POS tags
wnl = WordNetLemmatizer()
tagged = pos_tag(tokens_no_stop)

def to_wn(pos_tag):
    return {"J":"a","V":"v","N":"n","R":"r"}.get(pos_tag[0], "n")

lemmas = [wnl.lemmatize(w, to_wn(p)) for w,p in tagged]
print("Lemmas:", lemmas)

