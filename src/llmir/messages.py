from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from .chunks import AIChunks, AIChunkText
from .roles import AIRoles

class AIMessage(BaseModel):
    """
    A message in the LLM conversation.

    Attributes:
        role: The role of the message sender.
        chunks: The content chunks.
    """

    role: Literal[AIRoles.USER, AIRoles.MODEL, AIRoles.SYSTEM]
    chunks: list[AIChunks] = Field(default_factory=list[AIChunks])

    @classmethod
    def text(cls,
        text: str,
        role: Literal[AIRoles.USER, AIRoles.MODEL, AIRoles.SYSTEM],
    ) -> AIMessage:
        return AIMessage(
            role=role,
            chunks=[
                AIChunkText(text=text)
            ]
        )
        

class AIMessageToolResponse(BaseModel):
    """
    A special tool response message in the LLM conversation.

    Attributes:
        role: The role of the message sender.
        chunks: The content chunks.
        id: The id of the tool.
        name: The name of the tool.
    """

    role: Literal[AIRoles.TOOL] = AIRoles.TOOL
    chunks: list[AIChunks] = Field(default_factory=list[AIChunks])
    id: str
    name: str

AIMessages = AIMessage | AIMessageToolResponse