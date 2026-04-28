"""Text splitter module."""
import re
from typing import List, Tuple
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


class MarkdownHierarchicalSplitter:
    """
    按 Markdown 标题层级结构切分文档，每个 chunk 保留完整的标题路径。

    例如对于如下结构：
        # Title
        ## Section A
        ### Sub A1
        content...
        ### Sub A2
        content...

    每个 chunk 会保留完整路径：
        # Title
        ## Section A
        ### Sub A1
        content...
    """

    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')

    def get_nodes_from_documents(self, documents: List[Document]) -> List[TextNode]:
        """将文档列表切分为层级化的 chunks"""
        all_nodes = []
        for doc in documents:
            nodes = self.split_markdown(doc.text, doc.metadata)
            all_nodes.extend(nodes)
        return all_nodes

    def split_markdown(self, text: str, metadata: dict = None) -> List[TextNode]:
        """
        解析 Markdown 文本，按标题层级切分。

        Args:
            text: Markdown 文本
            metadata: 文档元数据

        Returns:
            TextNode 列表，每个包含标题路径和内容
        """
        lines = text.split('\n')
        chunks = []
        current_path: List[Tuple[int, str]] = []  # [(level, title), ...]
        current_content_lines: List[str] = []
        doc_started = False

        def emit_chunk():
            """输出当前累积的 chunk（只有包含实际内容时才输出）"""
            if not current_content_lines:
                return
            # 构建 chunk 文本：标题路径 + 内容
            chunk_lines = []
            for level, title in current_path:
                chunk_lines.append('#' * level + ' ' + title)
            chunk_lines.append('')
            chunk_lines.extend(current_content_lines)
            chunk_text = '\n'.join(chunk_lines)
            chunks.append(chunk_text)

        for line in lines:
            header_match = self.HEADER_PATTERN.match(line)

            if header_match:
                # 遇到新标题，先输出上一个 chunk
                if current_path and current_content_lines:
                    emit_chunk()

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # 更新标题路径：保留比当前级别更具体的子标题
                current_path = current_path[:level - 1]
                current_path.append((level, title))
                current_content_lines = []
                doc_started = True

            elif doc_started or line.strip():
                # 非标题行，累积内容（忽略纯空白行）
                if doc_started or current_path:
                    if line.strip():  # 只累积非空白行
                        current_content_lines.append(line)

        # 输出最后一个 chunk
        if current_path or current_content_lines:
            emit_chunk()

        # 转换为 TextNode
        nodes = []
        for i, chunk_text in enumerate(chunks):
            node = TextNode(
                text=chunk_text,
                metadata=metadata.copy() if metadata else {}
            )
            node.metadata['chunk_index'] = i
            nodes.append(node)

        return nodes


def split_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    parser_type: str = "sentence",
) -> List[TextNode]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of documents to split
        chunk_size: Target size of each chunk (in characters, approximately)
        chunk_overlap: Number of overlapping characters between chunks
        parser_type: Parser type - "sentence" (SentenceSplitter) or "markdown" (MarkdownHierarchicalSplitter)

    Returns:
        List of chunked nodes
    """
    if parser_type == "markdown":
        splitter = MarkdownHierarchicalSplitter()
        return splitter.get_nodes_from_documents(documents)
    else:
        parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return parser.get_nodes_from_documents(documents)


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split raw text into chunks.

    Args:
        text: Raw text to split
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Create a temporary document and split it
    doc = Document(text=text)
    nodes = parser.get_nodes_from_documents([doc])
    return [node.text for node in nodes]
