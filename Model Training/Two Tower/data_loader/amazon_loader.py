import json
import os
import pandas as pd
from features.sentiment_quality import ReviewSentimentAnalyzer

class AmazonDataLoader:
    def __init__(self, base_path="/Users/rohitbhatia/EE6405_NLP_RecommedationSystem/"):
        self.base_path = base_path

    def load_jsonl(self, path):
        with open(path) as f:
            return pd.DataFrame([json.loads(l) for l in f if l.strip()])

    def load_beauty_data(self):
        reviews = os.path.join(self.base_path, "All_Beauty.jsonl")
        meta = os.path.join(self.base_path, "meta_All_Beauty.jsonl")
        print("Loading Amazon Beauty data...")
        return self.load_jsonl(reviews), self.load_jsonl(meta)

