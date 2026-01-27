from pydantic import BaseModel, Field
from typing import Literal
from .chunks import AIChunk
from .roles import AIRoles

class AIMessage(BaseModel):

    role: Literal[AIRoles.USER, AIRoles.MODEL, AIRoles.SYSTEM]
    chunks: list[AIChunk] = Field(default_factory=list[AIChunk])

class AIMessageToolResponse(BaseModel):

    role: Literal[AIRoles.TOOL] = AIRoles.TOOL
    chunks: list[AIChunk] = Field(default_factory=list[AIChunk])
    id: str
    name: str

AIMessages = AIMessage | AIMessageToolResponse