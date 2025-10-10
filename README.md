# Twitter Topic Analysis

Dieses Projekt analysiert Tweets für eine Stadt/Region, bereinigt die Texte, extrahiert Hashtags & User
und bestimmt die häufigsten Themen mittels intelligenter Term-Filterung und Coherence Score.

## Setup

```bash
# 1) Umgebung erstellen
python -m venv .venv
.venv/bin/activate

# 2) Abhängigkeiten installieren
pip install -r requirements.txt

# 3) Credentials
cp .env.example .env
# → TWITTER_BEARER_TOKEN eintragen

# 4) Sprachressourcen
python -m spacy download de_core_news_sm
python -c "import nltk; import nltk; nltk.download('stopwords')"

# 5) Ausführen (max_tweets 100, wegen X Dev Free Plan)
./fetch_and_analyze.py --query "(Berlin OR #Berlin) lang:de -is:retweet" --max_tweets 100 --out_dir outputs
