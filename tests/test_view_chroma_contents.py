"""查看 ChromaDB 全部内容"""
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import chromadb
from chromadb.config import Settings

from src.config.settings import CHROMA_DB_DIR, KNOWLEDGE_BASE_DIR

# 连接 ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

print("=" * 60)
print("ChromaDB 内容查看")
print("=" * 60)

# 列出所有 collection
collections = client.list_collections()
print(f"\n[1] Collection 列表 (共 {len(collections)} 个):")
for col in collections:
    print(f"   - {col.name}")

print(f"\n[2] ChromaDB 存储目录: {CHROMA_DB_DIR}")
print(f"    目录是否存在: {CHROMA_DB_DIR.exists()}")

if CHROMA_DB_DIR.exists():
    files = list(CHROMA_DB_DIR.glob("*"))
    print(f"    文件列表 (共 {len(files)} 个):")
    for f in files:
        size = f.stat().st_size
        print(f"   - {f.name} ({size / 1024:.1f} KB)")

# 查看默认 collection 的内容
collection_name = "research_assistant"
print(f"\n[3] Collection '{collection_name}' 详细内容:")

try:
    collection = client.get_collection(name=collection_name)

    # 总数量
    count = collection.count()
    print(f"   文档数量: {count}")

    if count > 0:
        # 获取所有数据
        results = collection.get(include=["documents", "metadatas", "embeddings"])

        print(f"\n   IDs ({len(results['ids'])} 个):")
        for i, id_ in enumerate(results['ids']):
            print(f"      [{i}] {id_}")

        print(f"\n   Metadatas ({len(results['metadatas'])} 个):")
        for i, meta in enumerate(results['metadatas']):
            print(f"      [{i}] {meta}")

        print(f"\n   Documents 内容:")
        for i, doc in enumerate(results['documents']):
            print(f"   --- Document {i} ---")
            # 只显示前200字符
            display = doc[:200].replace('\n', ' ')
            print(f"   {display}..." if len(doc) > 200 else f"   {display}")

        # Embeddings 维度
        if results['embeddings'] and results['embeddings'][0]:
            dim = len(results['embeddings'][0])
            print(f"\n   Embeddings 维度: {dim}")

except Exception as e:
    print(f"   错误: {e}")

print("\n" + "=" * 60)
