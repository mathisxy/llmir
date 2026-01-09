from dataclasses import dataclass


@dataclass(frozen=True)
class AIChunk:
    pass

@dataclass(frozen=True)
class AIChunkText(AIChunk):

    content: str


@dataclass(frozen=True)
class AIChunkFile(AIChunk):

    name: str
    mimetype: str
    content: bytes