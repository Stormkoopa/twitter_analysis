import os
import re
import argparse
from collections import Counter
from typing import List, Tuple
import pandas as pd
import numpy as np
import tweepy
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser

# Setup stopwords and regex patterns
nltk.download("stopwords", quiet=True)
STOP_DE = set(stopwords.words("german")) | {"rt", "amp", "https", "co"}

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
NON_WORD_RE = re.compile(r"[^a-zA-ZäöüÄÖÜß0-9\s]")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    # Clean tweet text by removing URLs/mentions/hashtags/special characters/extra whitespace

    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = s.replace("#", " ")
    s = s.lower()
    s = NON_WORD_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def tokenize_basic(s: str) -> List[str]:
    # Basic tokenization
    return [t for t in s.split() if t not in STOP_DE and len(t) > 2 and not t.isdigit()]


def build_phrases(token_lists: List[List[str]], min_count=8, threshold=8.0) -> List[List[str]]:
    # Detect frequent bigrams and trigrams using Gensim's Phrases model
    bigram = Phrases(token_lists, min_count=min_count, threshold=threshold)
    trigram = Phrases(bigram[token_lists], min_count=min_count, threshold=threshold)
    big = Phraser(bigram)
    tri = Phraser(trigram)
    return [tri[big[tl]] for tl in token_lists]


def tfidf_filter(token_lists: List[List[str]], keep_percentile: int = 20) -> List[List[str]]:
    
    # Apply global TF-IDF filtering:
    # - calculate average TF-IDF weight for each token
    # - keep only tokens above a percentile cutoff (e.g. 20 keeps top 80%)
    docs = [" ".join(tl) for tl in token_lists]
    v = TfidfVectorizer(min_df=3, max_df=0.6)
    X = v.fit_transform(docs)
    means = X.mean(axis=0).A1
    vocab = v.get_feature_names_out()
    cutoff = np.percentile(means, keep_percentile)
    keep = {w for w, sc in zip(vocab, means) if sc >= cutoff}
    return [[w for w in tl if w in keep] for tl in token_lists]


def fetch_tweets(client, query: str, max_tweets: int = 400) -> Tuple[pd.DataFrame, dict]:
    # Fetch tweets using Tweepy with Twitter API v2
    rows, user_map = [], {}
    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        max_results=100,
        tweet_fields=["id", "text", "author_id", "created_at", "lang", "public_metrics"],
        user_fields=["id", "name", "username"],
        expansions=["author_id"]
    )
    for resp in paginator:
        if resp and resp.includes and "users" in resp.includes:
            for u in resp.includes["users"]:
                user_map[u.id] = {"username": u.username, "name": u.name}
        if resp and resp.data:
            for t in resp.data:
                rows.append({
                    "id": t.id,
                    "text": t.text,
                    "author_id": t.author_id,
                    "created_at": t.created_at,
                    "lang": t.lang,
                    "retweet_count": (t.public_metrics or {}).get("retweet_count", 0),
                    "reply_count": (t.public_metrics or {}).get("reply_count", 0),
                    "like_count": (t.public_metrics or {}).get("like_count", 0),
                    "quote_count": (t.public_metrics or {}).get("quote_count", 0),
                })
                if len(rows) >= max_tweets:
                    return pd.DataFrame(rows), user_map
    return pd.DataFrame(rows), user_map


def extract_entities(df: pd.DataFrame, user_map: dict):
    # Extract most frequent hashtags and most active users from tweets
    hashtags, users = Counter(), Counter()
    for _, r in df.iterrows():
        for h in re.findall(r"#(\w+)", r["text"]):
            hashtags[h.lower()] += 1
        users[r["author_id"]] += 1
        if pd.notna(aid):
            users[aid] += 1
    df_hashtags = pd.DataFrame([{"hashtag": h, "count": c} for h, c in hashtags.most_common()])
    df_users = pd.DataFrame([{
        "author_id": aid,
        "username": user_map.get(aid, {}).get("username", str(aid)),
        "name": user_map.get(aid, {}).get("name", ""),
        "count": c
    } for aid, c in users.most_common()])
    return df_hashtags, df_users


def pick_best_lda(token_lists: List[List[str]], k_grid=range(3, 11), passes=8, random_state=42):
    # Train LDA models with different numbers of topics and evaluate coherence (c_v)
    dictionary = Dictionary(token_lists)
    dictionary.filter_extremes(no_below=3, no_above=0.6)
    corpus = [dictionary.doc2bow(tl) for tl in token_lists]

    results = []
    best = {"k": None, "coh": -1.0, "model": None, "dictionary": dictionary}
    for k in k_grid:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=passes, random_state=random_state)
        cm = CoherenceModel(model=lda, texts=token_lists, dictionary=dictionary, coherence='c_v')
        coh = cm.get_coherence()
        results.append((k, coh))
        if coh > best["coh"]:
            best.update({"k": k, "coh": coh, "model": lda})
    return best, results, dictionary, corpus


def lda_top_terms(lda: LdaModel, topn=8):
    # Return the top n terms for each topic of the LDA model.
    return [{"topic": tid, "top_terms": ", ".join([w for w, p in lda.show_topic(tid, topn=topn)])}
            for tid in range(lda.num_topics)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default='(Berlin OR #Berlin) lang:de -is:retweet')
    parser.add_argument("--max_tweets", type=int, default=100)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--keep_percentile", type=int, default=20)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load or fetch tweets (Limited by X Dev Free Plan)
    if args.reuse and os.path.exists(os.path.join(args.out_dir, "tweets_raw.csv")):
        df_raw = pd.read_csv(os.path.join(args.out_dir, "tweets_raw.csv"))
        user_map = {}
    else:
        load_dotenv()
        bearer = os.getenv("TWITTER_BEARER_TOKEN")
        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
        df_raw, user_map = fetch_tweets(client, args.query, args.max_tweets)
        df_raw.to_csv(os.path.join(args.out_dir, "tweets_raw.csv"), index=False)

    # Preprocess
    df = df_raw.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df.to_csv(os.path.join(args.out_dir, "tweets_clean.csv"), index=False)

    # Entities
    df_hashtags, df_users = extract_entities(df_raw, user_map)
    df_hashtags.to_csv(os.path.join(args.out_dir, "hashtags_top.csv"), index=False)
    df_users.to_csv(os.path.join(args.out_dir, "users_top.csv"), index=False)

    # Tokenization/Filtering
    tokens = [tokenize_basic(t) for t in df["text_clean"].tolist()]
    tokens = [tl for tl in tokens if tl]
    tokens = build_phrases(tokens)
    tokens = tfidf_filter(tokens, keep_percentile=args.keep_percentile)
    tokens = [tl for tl in tokens if tl]

    # Topic Modeling with Coherence
    best, grid, dictionary, corpus = pick_best_lda(tokens, k_grid=range(3, 11), passes=8)
    lda = best["model"]
    topics = lda_top_terms(lda, topn=8)

    # Save topics
    topics_df = pd.DataFrame(topics).sort_values("topic").reset_index(drop=True)
    topics_df.to_csv(os.path.join(args.out_dir, "topics_top.csv"), index=False)

    # Save coherence scores
    coh_df = pd.DataFrame([{"k": k, "coherence_c_v": coh} for k, coh in grid]).sort_values("k")
    coh_df.to_csv(os.path.join(args.out_dir, "coherence_scores.csv"), index=False)

if __name__ == "__main__":
    main()
