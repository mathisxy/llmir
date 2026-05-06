from abc import ABC, abstractmethod

from ..messages import AIMessages, AIChunks
from ..tools import AITool


class BaseAdapter(ABC):
    """
    Base class for all adapters to other LLM providers.
    """

    @classmethod
    @abstractmethod
    def chat(cls, messages: list[AIMessages]) -> ...:
        """
        Adapter for AIMessage lists to the specific provider's format.
        """
        ...

    @classmethod
    @abstractmethod
    def tools(cls, tools: list[AITool]) -> ...:
        """
        Adapter for AITool lists to the specific provider's format.
        """
        ...


class UnsupportedChunk(Exception):
    """
    Exception raised when an unsupported chunk is encountered.

    Attributes:
        chunk: The unsupported chunk.
        message: Explanation of the error.
    """
    def __init__(self, chunk: AIChunks, message: str | None = None):
        self.chunk = chunk
        super().__init__(message or f"Unsupported chunk type: {getattr(chunk, 'type', chunk)}")