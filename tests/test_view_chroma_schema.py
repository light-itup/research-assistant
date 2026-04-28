"""查看 ChromaDB Schema 结构"""
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import chromadb

from src.config.settings import CHROMA_DB_DIR

# 连接 ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

print("=" * 60)
print("ChromaDB Schema 结构查看")
print("=" * 60)

# 列出所有 collection 及其 schema
collections = client.list_collections()
print(f"\n[1] Collection Schema 详情:")

for col in collections:
    print(f"\n   Collection: {col.name}")
    print(f"   Metadata: {col.metadata}")

    try:
        collection = client.get_collection(name=col.name)
        count = collection.count()
        print(f"   文档数: {count}")

        # 获取一条样本看结构
        if count > 0:
            sample = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])

            print(f"\n   Sample ID: {sample['ids'][0]}")
            print(f"\n   Metadata keys: {list(sample['metadatas'][0].keys()) if sample['metadatas'] else 'N/A'}")

            doc = sample['documents'][0]
            print(f"\n   Document length: {len(doc)} chars")
            print(f"   Document preview: {doc[:150].replace(chr(10), ' ')}...")

            if sample['embeddings'] and sample['embeddings'][0]:
                emb = sample['embeddings'][0]
                print(f"\n   Embedding dimension: {len(emb)}")
                print(f"   Embedding type: {type(emb[0]).__name__}")
                print(f"   Embedding preview (first 5 values): {emb[:5]}")

    except Exception as e:
        print(f"   错误: {e}")

# 尝试用 Peek 方法获取样本
print("\n" + "=" * 60)
print("[2] 使用 peek() 获取样本 (更简洁的方式):")
print("=" * 60)

try:
    collection = client.get_collection(name="research_assistant")
    result = collection.peek(limit=3)

    print(f"\n   IDs: {result['ids']}")
    print(f"   Documents count: {len(result['documents'])}")
    for i, doc in enumerate(result['documents']):
        print(f"\n   --- Doc {i} ---")
        print(f"   {doc[:200].replace(chr(10), ' ')}...")

except Exception as e:
    print(f"   错误: {e}")

# 验证数据与源文档的对应关系
print("\n" + "=" * 60)
print("[3] 验证: 文档 chunks 与原始文件的对应关系:")
print("=" * 60)

try:
    collection = client.get_collection(name="research_assistant")
    results = collection.get(include=["documents", "metadatas"])

    print(f"\n   共 {len(results['ids'])} 个 chunks\n")

    for i, (id_, doc, meta) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        print(f"   Chunk {i}:")
        print(f"     ID: {id_}")
        print(f"     Source file: {meta.get('file_name', 'unknown')}")
        print(f"     Doc length: {len(doc)} chars")

        # 显示与原文的对应关系
        preview = doc[:100].replace('\n', ' ')
        print(f"     Content: {preview}...")
        print()

except Exception as e:
    print(f"   错误: {e}")

print("=" * 60)
