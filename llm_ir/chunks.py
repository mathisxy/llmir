from typing import Union, Literal

from pydantic import BaseModel

class AIChunkText(BaseModel):
    type: Literal["text"] = "text"
    content: str

class AIChunkFile(BaseModel):
    type: Literal["file"] = "file"
    name: str
    mimetype: str
    content: bytes

AIChunk = Union[AIChunkText, AIChunkFile]