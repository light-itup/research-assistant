"""测试 MarkdownHierarchicalSplitter 层级化切分效果"""
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.rag.text_splitter import split_documents, MarkdownHierarchicalSplitter
from src.rag import load_documents
from src.config.settings import KNOWLEDGE_BASE_DIR

print("=" * 70)
print("对比: SentenceSplitter vs MarkdownHierarchicalSplitter")
print("=" * 70)

# 加载 Markdown 文档
docs = load_documents(
    directory=str(KNOWLEDGE_BASE_DIR),
    files=[str(KNOWLEDGE_BASE_DIR / "transformer_intro.md")]
)
print(f"\n加载文档: {docs[0].metadata.get('file_name')}")
print(f"原文长度: {len(docs[0].text)} 字符")

# 方案1: SentenceSplitter (默认)
print("\n" + "-" * 50)
print("方案1: SentenceSplitter (按字符数切分)")
print("-" * 50)

nodes1 = split_documents(docs, parser_type="sentence")
print(f"切分成 {len(nodes1)} 个 chunks:")

for i, node in enumerate(nodes1):
    preview = node.text[:80].replace('\n', ' ')
    print(f"  [{i}] ({len(node.text)} chars) {preview}...")

# 方案2: MarkdownHierarchicalSplitter (按标题层级切分)
print("\n" + "-" * 50)
print("方案2: MarkdownHierarchicalSplitter (按标题层级切分)")
print("-" * 50)

nodes2 = split_documents(docs, parser_type="markdown")
print(f"切分成 {len(nodes2)} 个 chunks:")

for i, node in enumerate(nodes2):
    preview = node.text[:80].replace('\n', ' ')
    print(f"  [{i}] ({len(node.text)} chars) {preview}...")

# 详细查看每个 chunk 的内容
print("\n" + "=" * 70)
print("MarkdownHierarchicalSplitter 详细结果")
print("=" * 70)

for i, node in enumerate(nodes2):
    print(f"\n--- Chunk {i} ---")
    print(node.text)
    print("---")

print("=" * 70)