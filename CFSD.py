DATASET_PATH = "D:\\yelp_sentiment_master_dataset" 
 
USE_MYSQL = False          
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "your_password",   
    "database": "yelp_restaurant"  
}
import pandas as pd
import re
import os
import datetime
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
 
warnings.filterwarnings("ignore")
 
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it",
    "its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","just","would","could","also","get","got","really","quite",
    "bit","came","come","went","go","one","two","will","can","still","even",
    "back","much","well","now","like","make","way","say","said","us","told",
    "upon","every","never","always","often","time","times","many","around",
    "another","first","second","three","four","five","six","seven","eight",
    "nine","ten","however","though","although","since","without","whether",
    "something","anything","everything","nothing","someone","anyone","everyone",
    "place","places","going","came","want","wanted","asked","table","tables",
    "been","has","had","took","take","made","new","old","little","big","small",
    "see","look","looks","looked","think","thought","know","knew","lot","lots",
    "pretty","kind","sure","try","tried","tried","need","tried","put","long"
}

POSITIVE_KEYWORDS = {
    "excellent": 15, "outstanding": 15, "amazing": 12, "perfect": 12,
    "best":      10, "delicious":   10, "wonderful": 10, "fantastic": 10,
    "great":      8, "good":         6, "fresh":      5, "tasty":      5,
    "recommend":  8, "loved":        8, "divine":    10, "incredible": 12,
    "impeccable":10, "friendly":     5, "attentive":  5, "superb":    12,
    "brilliant":  10,"enjoyable":    8, "pleasant":   6, "awesome":   10,
    "fabulous":  10, "lovely":        7, "satisfied":  6, "happy":      5,
    "favorite":   8, "love":          7, "nice":       4, "solid":      5,
    "impressed":  8, "yummy":         8, "flavorful":  8, "tender":     6,
    "crispy":     5, "creamy":        5, "authentic":  7, "cozy":       5,
}
 
NEGATIVE_KEYWORDS = {
    "cold":       -10, "terrible":   -15, "disgusting": -15, "horrible": -15,
    "worst":      -15, "rude":       -12, "bad":        -10, "disappointing":-12,
    "awful":      -13, "stale":       -8, "soggy":       -8, "wrong":     -8,
    "slow":        -6, "waited":      -6, "refund":      -8, "hair":     -15,
    "undercooked":-12, "tasteless":  -10, "waste":      -10, "mediocre":  -7,
    "bland":       -8, "overpriced": -10, "dirty":      -12, "disgusted": -12,
    "awful":      -13, "gross":      -10, "burnt":       -9, "dry":        -6,
    "unfriendly":  -9, "inattentive": -8, "freezing":    -8, "lousy":    -10,
    "nasty":      -12, "unacceptable":-12,"poor":        -8, "disappointed":-10,
    "overrated":   -9, "inconsistent":-7, "unprofessional":-10,
}
 
 
def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)
 
 
def clean_text(text: str) -> str:
   
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_emojis(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)
 
 # test score 
def calculate_sentiment_score(row) -> float:
    base = (row["stars_review"] / 5.0) * 60.0
    adj  = sum(
        POSITIVE_KEYWORDS.get(w, 0) + NEGATIVE_KEYWORDS.get(w, 0)
        for w in str(row["cleaned_text"]).split()
    )
    return round(max(0.0, min(100.0, base + adj)), 1)
 
 
def classify_sentiment(score: float) -> str:
    if score >= 70:  return "Positive"
    if score >= 40:  return "Neutral"
    return "Negative"


 # daily sql export
def export_to_mysql(negative_df):
    try:
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
 
    
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS negative_feedback (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                review_id   INT,
                branch      VARCHAR(100),
                rating      INT,
                sentiment_score FLOAT,
                cleaned_text    TEXT,
                review_date     DATE,
                exported_at     DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        insert_query = """
            INSERT INTO negative_feedback
            (review_id, branch, rating, sentiment_score, cleaned_text, review_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        for _, row in negative_df.iterrows():
            cursor.execute(insert_query, (
                int(row["review_id"]),
                row["branch"],
                int(row["rating"]),
                float(row["sentiment_score"]),
                row["cleaned_text"],
                row["date"].date()
            ))
 
        conn.commit()
        print(f"  ✓ {len(negative_df)} negative feedback rows exported to MySQL.")
        cursor.close()
        conn.close()
 
    except ImportError:
        print("  ✗ mysql-connector-python not installed. Run: pip install mysql-connector-python")
    except Exception as e:
        print(f"  ✗ MySQL Error: {e}")
        print("    → Check your DB_CONFIG credentials and that MySQL is running.")
 