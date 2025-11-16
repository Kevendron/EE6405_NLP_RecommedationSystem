import numpy
import pandas as pd
from features.sentiment_quality import ReviewSentimentAnalyzer


class AmazonSequenceBuilder:
    def __init__(self, max_seq_len=20):
        self.max_seq_len = max_seq_len
        self.sa = ReviewSentimentAnalyzer()

    def _feat(self, r):
        txt = r.get('text', '')
        q = self.sa.calc_quality(txt, r.get('rating', 0), r.get('helpful_vote', 0), r.get('verified_purchase', False))
        L = min(len(str(txt))/200.0, 1.0)
        return [float(r.get('rating',0))/5.0, L,
                min(float(r.get('helpful_vote',0))/5.0,1.0),
                1.0 if r.get('verified_purchase',False) else 0.0,
                q]

    def build_sequences_from_amazon(self, df):
        df = df.dropna(subset=['user_id','parent_asin']).copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values(['user_id','timestamp'])
        active = df['user_id'].value_counts()
        df = df[df['user_id'].isin(active[active>=3].index)]

        seqs=[]
        for uid, grp in df.groupby('user_id'):
            grp = grp.tail(50)
            rows=list(grp.to_dict('records'))
            for i in range(1,len(rows)):
                ctx=rows[max(0,i-self.max_seq_len):i]
                tgt=rows[i]
                items=[r['parent_asin'] for r in ctx]
                feats=[self._feat(r) for r in ctx]
                while len(items)<self.max_seq_len:
                    items.insert(0,'PAD'); feats.insert(0,[0]*5)
                mask=[1 if it!='PAD' else 0 for it in items]
                seqs.append({'user_id':uid,'sequence_items':items[-self.max_seq_len:],
                             'quality_features':feats[-self.max_seq_len:],
                             'sequence_mask':mask,'target_item':tgt['parent_asin']})
        print(f"Built {len(seqs)} sequences from {df['user_id'].nunique()} users")
        return seqs
    
def create_item_texts_from_metadata(df):
    item_texts = {}
    for _, row in df.iterrows():
        asin = row.get('parent_asin') or row.get('asin')
        if not isinstance(asin, str): continue
        parts = []
        if isinstance(row.get('title'), str): parts.append(f"Product: {row['title']}")
        if isinstance(row.get('brand'), str): parts.append(f"Brand: {row['brand']}")
        if isinstance(row.get('categories'), list):
            parts.append("Category: " + " > ".join(map(str, row['categories'][:3])))
        desc = row.get('description')
        if isinstance(desc, list): desc = " ".join(desc[:2])
        if isinstance(desc, str): parts.append(f"Description: {desc[:300]}")
        if parts: item_texts[asin] = ". ".join(parts)[:1500]
    print(f"Created texts for {len(item_texts)} items")
    return item_texts