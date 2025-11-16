from transformers import RobertaTokenizer
from utils.device import get_device
from models.ItemEmbedding import ItemEmbeddingModel
from models.UserEmbeddings import UserEmbeddingModel
from models.CombinedEmbedding import CombinedEmbeddingModel
from training.trainer import train_model
from evaluation.metrics import evaluate_recall
from data_loader.amazon_loader import AmazonDataLoader
from data_loader.sequence_builder import AmazonSequenceBuilder
from config import *
from data_loader.sequence_builder import create_item_texts_from_metadata
from sklearn.model_selection import train_test_split

def main():
    device = get_device()
    print(device)
    tokenizer=RobertaTokenizer.from_pretrained('roberta-base')
    item_model=ItemEmbeddingModel().to(device)
    user_model=UserEmbeddingModel().to(device)
    model=CombinedEmbeddingModel(item_model,user_model).to(device)

    loader=AmazonDataLoader()
    reviews,meta=loader.load_beauty_data()
    item_texts=create_item_texts_from_metadata(meta)
    builder=AmazonSequenceBuilder(max_seq_len=20)
    seqs=builder.build_sequences_from_amazon(reviews)
    train_seqs,val_seqs=train_test_split(seqs,test_size=0.2,random_state=42)

    train_model(model, train_seqs, item_texts, tokenizer, device, bs=8, epochs=5)


if __name__=="__main__":
    main()
