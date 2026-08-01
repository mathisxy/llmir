from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from .chunks import AIChunkFile, AIChunkImageURL, AIChunkToolCall, AIChunks, AIChunkText
from .roles import AIRoles

class AIMessage(BaseModel):
    """
    A message in the LLM conversation.

    Attributes:
        role: The role of the message sender.
        chunks: The content chunks.
    """

    Roles = AIRoles

    class Chunk:
        Text = AIChunkText
        File = AIChunkFile
        ImageURL = AIChunkImageURL
        ToolCall = AIChunkToolCall

        Any = AIChunks

    role: Literal[AIRoles.USER, AIRoles.MODEL, AIRoles.SYSTEM]
    chunks: list[Chunk.Any] = Field(default_factory=list["Chunk.Any"])


    @classmethod
    def text(cls,
        text: str,
        role: Literal[Roles.USER, Roles.MODEL, Roles.SYSTEM],
    ) -> AIMessage:
        return AIMessage(
            role=role,
            chunks=[
                cls.Chunk.Text(text=text)
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