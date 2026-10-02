"""Download and process Brazilian Customer Service Conversations dataset from Hugging Face."""

import json
import os
import urllib.request
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

URL = "https://huggingface.co/datasets/RichardSakaguchiMS/brazilian-customer-service-conversations/resolve/main/data/conversations.jsonl"
OUTPUT_FILE = os.path.join(RAW_DIR, "conversas_ptbr.csv")

# Map english sentiment from dataset to portuguese
SENTIMENT_MAP = {
    'positive': 'positivo',
    'neutral': 'neutro',
    'negative': 'negativo'
}

def download_and_prepare():
    print(f"Baixando dataset de: {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    
    records = []
    with urllib.request.urlopen(req) as response:
        for line in response:
            line_str = line.decode('utf-8').strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            
            messages = item.get('messages', [])
            # Agrupa todas as falas do cliente para contextualizar a demanda
            customer_messages = [
                m['content'].strip() for m in messages if m.get('role') == 'customer'
            ]
            full_customer_text = " ".join(customer_messages)
            
            metadata = item.get('metadata', {})
            raw_sentiment = metadata.get('sentiment', 'neutral').lower()
            sentiment = SENTIMENT_MAP.get(raw_sentiment, 'neutro')
            
            intent = metadata.get('intent', '').strip().lower()
            sector = metadata.get('sector', '').strip().lower()
            
            # Monta tags multi-label compostas por intenção e setor
            tags_list = []
            if intent:
                tags_list.append(intent)
            if sector:
                tags_list.append(sector)
            tags_str = "|".join(tags_list)
            
            if full_customer_text:
                records.append({
                    'id': item.get('id', ''),
                    'text': full_customer_text,
                    'sentiment': sentiment,
                    'tags': tags_str,
                    'intent': intent,
                    'sector': sector,
                    'turns': metadata.get('turns', len(messages))
                })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Dataset salvo com sucesso em: {OUTPUT_FILE}")
    print(f"Total de registros: {len(df)}")
    print("\nDistribuição de sentimentos:")
    print(df['sentiment'].value_counts())
    print("\nDistribuição de intenções (tags):")
    print(df['intent'].value_counts().head(10))
    print("\nExemplo do primeiro registro:")
    print(df.iloc[0].to_dict())

if __name__ == '__main__':
    download_and_prepare()
