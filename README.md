# Twitter Topic Analysis

Dieses Projekt analysiert Tweets für eine Stadt/Region, bereinigt die Texte, extrahiert Hashtags & User
und bestimmt die häufigsten Themen mittels intelligenter Term-Filterung und Coherence Score.

## Kurze technische Beschreibung zur Nutzung des Codes:
Für die Ausführung wird eine aktuelle Python-3-Version benötigt (empfohlen: Python ≥ 3.9) sowie die Installation aller im Repository angegebenen Abhängigkeiten aus der Datei requirements.txt:
- tweepy
- pandas
- nltk
- gensim
- scikit-learn
- python-dotenv

Die Einrichtung einer virtuellen Umgebung wird dringend empfohlen, um Versionskonflikte zu vermeiden.

Vor dem Start muss im Projektverzeichnis eine .env-Datei erstellt werden, die den persönlichen Twitter-Bearer-Token enthält. Dieser wird für die Authentifizierung gegenüber der Twitter-API benötigt und hat folgendes Format:
TWITTER_BEARER_TOKEN=%YOUR_TWITTER_BEARER_TOKEN%

Das Hauptskript fetch_and_analyze.py kann anschließend über die Kommandozeile gestartet werden. Dabei können optionale Parameter wie die Suchanfrage (--query), die maximale Anzahl der Tweets (--max_tweets), der Ausgabepfad (--out_dir) oder die Wiederverwendung bereits gespeicherter Tweets (--reuse) übergeben werden. Beispiel:

python fetch_and_analyze.py --query "(Berlin OR #Berlin) lang:de -is:retweet" --max_tweets 100

Nach der Ausführung werden die Ergebnisse im angegebenen Ausgabeverzeichnis abgelegt. Dazu gehören unter anderem:
-	tweets_raw.csv – die rohen, von der API geladenen Tweets
-	tweets_clean.csv – bereinigte und vorverarbeitete Tweet-Texte
-	hashtags_top.csv – die häufigsten Hashtags
-	users_top.csv – die aktivsten Nutzer
-	topics_top.csv – die identifizierten Themen mit ihren Top-Terms
-	coherence_scores.csv – Übersicht der berechneten Coherence Scores für verschiedene Topic-Zahlen
Diese Dateien können im Anschluss zur weiterführenden Analyse, Visualisierung oder Dokumentation genutzt werden.


## Setup

```bash
# 1) Virtuelle Umgebung erstellen
python -m venv .venv
.venv/bin/activate

# 2) Abhängigkeiten installieren
pip install -r requirements.txt

# 3) Ausführen (max_tweets 100, wegen X Dev Free Plan)
./fetch_and_analyze.py --query "(Berlin OR #Berlin) lang:de -is:retweet" --max_tweets 100 --out_dir outputs
