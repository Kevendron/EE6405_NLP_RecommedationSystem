import torch 
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)


class ReviewSentimentAnalyzer:
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
        self.cache = {}

    def analyze_review(self, text):
        text = str(text)
        if text in self.cache:
            return self.cache[text]
        if len(text) < 10:
            out = {"negative": 0.3, "neutral": 0.2, "positive": 0.5}
            self.cache[text] = out
            return out
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
        out = {"negative": probs[0].item(), "neutral": probs[1].item(), "positive": probs[2].item()}
        self.cache[text] = out
        return out

    def calc_quality(self, text, rating, helpful, verified):
        s = self.analyze_review(text)
        exp_pos = float(rating or 0)/5.0
        act_pos = s["positive"]
        consistency = 1 - abs(exp_pos - act_pos)
        length = min(len(str(text))/200.0, 1.0)
        helpful = min(float(helpful or 0)/5.0, 1.0)
        verified_bonus = 1.2 if verified else 1.0
        q = (consistency*0.4 + length*0.3 + helpful*0.3) * verified_bonus
        return float(min(q, 1.0))
    