"""构建持久化索引并查看内容"""
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import chromadb
from src.config.settings import CHROMA_DB_DIR, KNOWLEDGE_BASE_DIR
from src.rag import load_documents, create_embedder, create_vector_index

print("=" * 70)
print("步骤1: 构建持久化索引")
print("=" * 70)

# 加载文档
docs = load_documents(directory=str(KNOWLEDGE_BASE_DIR))
print(f"\n加载文档数: {len(docs)}")
for doc in docs:
    print(f"  - {doc.metadata.get('file_name', 'unknown')}")

# 创建 embedder
embedder = create_embedder()
print(f"\nEmbedder: {embedder.model_name}")

# 构建持久化索引 (store_locally=True)
print("\n构建持久化索引...")
index = create_vector_index(
    docs,
    embed_model=embedder,
    store_locally=True,
    persist_dir=str(CHROMA_DB_DIR),
    collection_name="research_assistant"
)
print(f"索引构建完成!")

print("\n" + "=" * 70)
print("步骤2: 查看 ChromaDB 内容")
print("=" * 70)

# 重新连接查看
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

# 列出所有 collection
collections = client.list_collections()
print(f"\n[1] Collection 列表 (共 {len(collections)} 个):")
for col in collections:
    print(f"   - {col.name}")

# 查看默认 collection 的内容
collection_name = "research_assistant"
collection = client.get_collection(name=collection_name)
count = collection.count()
print(f"\n[2] Collection '{collection_name}':")
print(f"   文档数量: {count}")

if count > 0:
    # 获取所有数据
    results = collection.get(include=["documents", "metadatas", "embeddings"])

    print(f"\n[3] IDs ({len(results['ids'])} 个):")
    for i, id_ in enumerate(results['ids']):
        print(f"      [{i}] {id_}")

    print(f"\n[4] 文件来源:")
    for i, meta in enumerate(results['metadatas']):
        fname = meta.get('file_name', 'unknown')
        print(f"      [{i}] {fname}")

    print(f"\n[5] Documents 内容 (共 {len(results['documents'])} 个 chunks):")
    for i, doc in enumerate(results['documents']):
        print(f"\n   --- Chunk {i} ---")
        print(f"   ID: {results['ids'][i]}")
        print(f"   长度: {len(doc)} 字符")
        print(f"   内容预览:")
        lines = doc.split('\n')
        for line in lines[:8]:
            if line.strip():
                print(f"      {line[:80]}")
        if len(lines) > 8:
            print(f"      ... (共 {len(lines)} 行)")

    # Embeddings 信息
    import numpy as np
    emb_list = results['embeddings']
    if emb_list is not None and len(emb_list) > 0 and emb_list[0] is not None:
        emb = np.array(emb_list[0])
        print(f"\n[6] Embeddings 信息:")
        print(f"   维度: {len(emb)}")
        print(f"   前5个值: {emb[:5]}")

print("\n" + "=" * 70)
print("步骤3: 验证向量检索功能")
print("=" * 70)

results = collection.query(
    query_texts=["self-attention mechanism"],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

print(f"\n查询: 'self-attention mechanism'")
print(f"返回结果数: {len(results['ids'][0])}")

for i in range(len(results['ids'][0])):
    print(f"\n   [结果 {i}]")
    print(f"   ID: {results['ids'][0][i]}")
    print(f"   距离: {results['distances'][0][i]:.4f}")
    doc_preview = results['documents'][0][i][:200].replace('\n', ' ')
    print(f"   文档预览: {doc_preview}...")

print("\n" + "=" * 70)
