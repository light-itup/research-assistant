"""
全局索引管理器 (Index Manager)

实现单例模式，管理向量索引的全生命周期。

使用方式:
    from src.rag.index_manager import IndexManager

    # 获取单例实例
    manager = IndexManager.get_instance()

    # 初始化（从 ChromaDB 加载已有索引）
    manager.initialize()

    # 查询（只向量化 query，不重复构建索引）
    results = manager.search("transformer architecture")

    # 检查状态
    if manager.is_ready():
        print("索引已就绪")
"""
from typing import List, Optional, Dict, Any

from src.rag.vector_store import load_existing_index, query_index
from src.rag.embedder import create_embedder


class IndexManager:
    """
    全局索引管理器（单例模式）

    负责:
    - 从 ChromaDB 加载已有索引
    - 提供查询接口
    - 管理索引生命周期
    """

    _instance: Optional['IndexManager'] = None

    def __init__(self):
        """初始化（私有方法，通过 get_instance 调用）"""
        self._embedder = None
        self._index = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> 'IndexManager':
        """
        获取 IndexManager 单例实例

        Returns:
            IndexManager 实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def embedder(self):
        """懒加载 embedder"""
        if self._embedder is None:
            self._embedder = create_embedder()
        return self._embedder

    def initialize(self) -> bool:
        """
        从 ChromaDB 加载已有索引

        如果索引不存在或加载失败，返回 False

        Returns:
            是否加载成功
        """
        if self._initialized:
            return self._index is not None

        try:
            self._index = load_existing_index(embed_model=self.embedder)
            self._initialized = True
            return True
        except Exception as e:
            print(f"[IndexManager] 索引加载失败: {e}")
            self._initialized = True
            return False

    def is_ready(self) -> bool:
        """
        检查索引是否已就绪

        Returns:
            索引是否可用
        """
        return self._index is not None

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        查询索引

        注意: 只对 query 进行向量化，不重复构建索引

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            filters: 元数据过滤器

        Returns:
            检索结果列表

        Raises:
            ValueError: 索引未初始化
        """
        # 自动初始化（如果尚未初始化）
        if not self.is_ready():
            if not self.initialize():
                raise ValueError(
                    "索引未初始化。请先运行: python scripts/init_knowledge_base.py"
                )

        return query_index(self._index, query, top_k=top_k, filters=filters)

    def reset(self):
        """
        重置管理器状态

        用于测试或重新初始化
        """
        self._index = None
        self._initialized = False

    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息

        Returns:
            包含索引信息的字典
        """
        if not self.is_ready():
            return {
                "ready": False,
                "message": "索引未初始化"
            }

        return {
            "ready": True,
            "index_type": type(self._index).__name__,
        }


# 全局便捷函数
def get_index_manager() -> IndexManager:
    """获取 IndexManager 单例（便捷函数）"""
    return IndexManager.get_instance()


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    搜索知识库（便捷函数）

    内部使用 IndexManager

    Args:
        query: 查询字符串
        top_k: 返回结果数量
        filters: 元数据过滤器

    Returns:
        检索结果列表
    """
    manager = IndexManager.get_instance()

    if not manager.is_ready():
        manager.initialize()

    return manager.search(query, top_k=top_k, filters=filters)
