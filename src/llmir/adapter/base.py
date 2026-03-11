from abc import ABC, abstractmethod

from ..messages import AIMessages
from ..tools import AITool

class BaseAdapter(ABC):

    @classmethod
    @abstractmethod
    def chat(cls, messages: list[AIMessages]) -> ...:
        ...

    @classmethod
    @abstractmethod
    def tools(cls, tools: list[AITool]) -> ...:
        ...