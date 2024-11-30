#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import tiktoken
from openai import OpenAI
import chromadb
from tqdm import tqdm
import time
import logging
import os
from pathlib import Path
from app.utils import read_json_file

CONFIG = read_json_file('app/config.json')
SECRETS = read_json_file('secrets.json')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self):
        self.client = OpenAI(api_key=SECRETS['openai_api_key'])
        self.model = CONFIG['embedding_model_name']
        self.chunk_size = CONFIG['chunk_size']
        self.chunk_overlap = CONFIG['chunk_overlap']
        self.batch_size = CONFIG['batch_size']
        self.max_retries = 3
        self.retry_delay = 1
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
            i += (self.chunk_size - self.chunk_overlap)
        return chunks

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts using OpenAI's latest API."""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == self.max_retries:
                        logger.error(f"Failed to get embeddings after {self.max_retries} retries: {str(e)}")
                        raise
                    logger.warning(f"Retry {retry_count}/{self.max_retries} after error: {str(e)}")
                    time.sleep(self.retry_delay * retry_count)
        return embeddings

    def process_csv(self, csv_path: str, text_columns: List[str], encoding: str = 'utf-8') -> List[Dict[str, Any]]:
        """Process CSV file and generate embeddings for specified text columns."""
        logger.info(f"Processing CSV file: {csv_path}")
        df = pd.read_csv(csv_path, encoding=encoding)
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            combined_text = " ".join([str(row[col]) for col in text_columns])
            chunks = self.chunk_text(combined_text)
            for chunk_idx, chunk in enumerate(chunks):
                results.append({
                    'text': chunk,
                    'source_idx': idx,
                    'chunk_idx': chunk_idx,
                    'metadata': {col: row[col] for col in df.columns}
                })
        texts = [item['text'] for item in results]
        embeddings = self.get_embeddings(texts)
        for idx, item in enumerate(results):
            item['embedding'] = embeddings[idx]
        return results

    def process_text_files(self, directory: str) -> List[Dict[str, Any]]:
        """Process all text files in a directory."""
        logger.info(f"Processing text files in directory: {directory}")
        results = []
        for file_path in Path(directory).glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = self.chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                results.append({
                    'text': chunk,
                    'source': str(file_path),
                    'chunk_idx': chunk_idx,
                    'metadata': {
                        'file_name': file_path.name,
                        'chunk_index': chunk_idx
                    }
                })
        texts = [item['text'] for item in results]
        embeddings = self.get_embeddings(texts)
        for idx, item in enumerate(results):
            item['embedding'] = embeddings[idx]
        return results

    def save_to_chromadb(self, data: List[Dict[str, Any]], collection_name: str, persist_directory: str = "./chroma_db"):
        """Save embeddings and metadata to ChromaDB."""
        persist_directory = os.path.abspath(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(f"ChromaDB data will be stored in: {persist_directory}")
        client = chromadb.PersistentClient(path=persist_directory)
        try:
            client.delete_collection(collection_name)
        except:
            pass
        collection = client.create_collection(name=collection_name)
        ids = [f"doc_{i}" for i in range(len(data))]
        embeddings = [item['embedding'] for item in data]
        documents = [item['text'] for item in data]
        metadatas = [item.get('metadata', {}) for item in data]
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info(f"Saved {len(data)} documents to ChromaDB collection '{collection_name}'")

if __name__ == "__main__":
    pipeline = EmbeddingPipeline()

    # Process CSV for low-level embeddings
    csv_results = pipeline.process_csv(
        csv_path="raw_data/reguleringsplan_enhanced.csv",
        text_columns=[
            'navn', 'ingress', 'beskrivelse', 'kontekstavhengig_beskrivelse',
            'normeringsniva', 'eif_niva', 'status', 'samhandlingstjenester',
            'ansvarlig', 'dokumenttype', 'referanse_lenketekst', 'referanse_url'
        ]
    )

    # Generate high-level embeddings
    df = pd.read_csv("raw_data/reguleringsplan_enhanced.csv", encoding='utf-8')
    grouped_data = df.groupby('informasjonstype').agg({
        'navn': lambda x: ' '.join(x.dropna()),
        'beskrivelse': lambda x: ' '.join(x.dropna())
    }).reset_index()
    grouped_data['combined_text'] = grouped_data['navn'] + " " + grouped_data['beskrivelse']
    grouped_data['embedding'] = pipeline.get_embeddings(grouped_data['combined_text'].tolist())
    high_level_results = [
        {
            'text': row['combined_text'],
            'embedding': row['embedding'],
            'metadata': {'category': row['informasjonstype']}
        }
        for _, row in grouped_data.iterrows()
    ]

    # Process text files
    #text_results = pipeline.process_text_files(directory="./raw_data")

    # Save all embeddings to ChromaDB
    pipeline.save_to_chromadb(csv_results, collection_name="low_level_embeddings")
    pipeline.save_to_chromadb(high_level_results, collection_name="high_level_embeddings")
    #pipeline.save_to_chromadb(text_results, collection_name="text_file_embeddings")