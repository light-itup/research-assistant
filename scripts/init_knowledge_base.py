"""
初始化知识库索引脚本

此脚本执行一次性操作：清空 ChromaDB 并重建索引

使用方法:
    python scripts/init_knowledge_base.py

执行时机:
    1. 首次使用知识库前
    2. 知识库内容发生变化后
"""
import os
import sys

# 设置 HF endpoint
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from src.rag import (
    load_documents,
    split_documents,
    create_vector_index,
)
from src.config.settings import CHROMA_DB_DIR, KNOWLEDGE_BASE_DIR


def clear_chroma_db():
    """清空 ChromaDB 中的所有 collections"""
    print("[1/5] 清空 ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collections = client.list_collections()

    for col in collections:
        client.delete_collection(name=col.name)
        print(f"    - 已删除 collection: {col.name}")

    print(f"    - ChromaDB 已清空 (共删除 {len(collections)} 个)")


def load_knowledge_base_documents():
    """加载知识库文档"""
    print("[2/5] 加载知识库文档...")
    docs = load_documents(directory=str(KNOWLEDGE_BASE_DIR))

    if not docs:
        print("    - 警告: 知识库为空!")
        return []

    for doc in docs:
        fname = doc.metadata.get('file_name', 'unknown')
        print(f"    - {fname} ({len(doc.text)} 字符)")

    print(f"    - 共加载 {len(docs)} 个文档")
    return docs


def split_documents_into_chunks(docs):
    """将文档切分为 chunks"""
    print("[3/5] 切分文档为 chunks...")

    # 使用 markdown 层级切分器
    chunks = split_documents(docs, parser_type="markdown")

    for i, chunk in enumerate(chunks):
        preview = chunk.text[:50].replace('\n', ' ')
        print(f"    - Chunk {i}: {preview}...")

    print(f"    - 共切分为 {len(chunks)} 个 chunks")
    return chunks


def build_vector_index(chunks):
    """构建向量索引并持久化"""
    print("[4/5] 构建向量索引...")

    index = create_vector_index(
        chunks,
        store_locally=True,
        persist_dir=str(CHROMA_DB_DIR),
        collection_name="research_assistant"
    )

    print("    - 索引构建完成!")
    return index


def verify_index():
    """验证索引构建成功"""
    print("[5/5] 验证索引...")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collections = client.list_collections()

    if not collections:
        print("    - 错误: 索引验证失败!")
        return False

    col = client.get_collection(name="research_assistant")
    count = col.count()
    print(f"    - Collection: {col.name}")
    print(f"    - 文档数量: {count}")
    print("    - 验证通过!")

    return True


def main():
    """主函数"""
    print("=" * 60)
    print("知识库索引初始化")
    print("=" * 60)
    print(f"知识库目录: {KNOWLEDGE_BASE_DIR}")
    print(f"向量数据库目录: {CHROMA_DB_DIR}")
    print()

    try:
        # 1. 清空
        clear_chroma_db()
        print()

        # 2. 加载文档
        docs = load_knowledge_base_documents()
        if not docs:
            print("\n错误: 知识库为空，请先添加文档到 data/knowledge_base/")
            return
        print()

        # 3. 切分
        chunks = split_documents_into_chunks(docs)
        print()

        # 4. 构建索引
        build_vector_index(chunks)
        print()

        # 5. 验证
        if verify_index():
            print()
            print("=" * 60)
            print("索引初始化完成!")
            print("=" * 60)
            print()
            print("下一步:")
            print("  1. 运行 Agent 测试: PYTHONPATH=. python -m tests.test_research_agent")
            print("  2. 如需重新初始化: python scripts/init_knowledge_base.py")
        else:
            print("\n索引验证失败，请检查错误信息。")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
