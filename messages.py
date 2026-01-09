from dataclasses import dataclass, field
from typing import List

from core.chunks import AIChunk
from core.roles import AIRoles


@dataclass(frozen=True)
class AIMessage:

    role: AIRoles
    chunks: List[AIChunk] = field(default_factory=list)