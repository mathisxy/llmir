from ollama import (
    Message,
    Tool,
    Image,
)

from ..messages import AIMessages, AIMessageToolResponse
from ..chunks import AIChunkText, AIChunkFile, AIChunkImageURL, AIChunkToolCall, AIChunks
from ..tools import AITool
from .base import BaseAdapter


class OllamaAdapter(BaseAdapter):


    @classmethod
    def parse_chunks(cls, chunks: list[AIChunks]) -> tuple[str, list[Image], list[Message.ToolCall]]:
        text = ""
        images: list[Image] = []
        tool_calls: list[Message.ToolCall] = []

        for chunk in chunks:

            match chunk:

                case AIChunkText():
                    text += chunk.text + "\n"

                case AIChunkImageURL():
                    images.append(Image(value=chunk.url))

                case AIChunkFile():
                    if chunk.mimetype.startswith("image/"):
                        images.append(Image(value=chunk.bytes))

                    elif chunk.mimetype == "text/plain":
                        text += chunk.bytes.decode("utf-8") + "\n"

                case AIChunkToolCall():
                    tool_calls.append(
                        Message.ToolCall(
                            function=Message.ToolCall.Function(
                                name=chunk.name,
                                arguments=chunk.arguments,
                            )
                        )
                    )

                case _:
                    raise ValueError(f"Unsupported chunk type: {type(chunk)}")

        return text, images, tool_calls

    @classmethod
    def chat(cls, messages: list[AIMessages]) -> list[Message]:

        result: list[Message] = []

        for message in messages:

            role = message.role.value
            text, images, tool_calls = cls.parse_chunks(message.chunks)

            msg = Message(
                role=role,
                content=text,
            )

            if isinstance(message, AIMessageToolResponse):
                msg.tool_name = message.name

            if images:
                msg.images = images

            if tool_calls:
                msg.tool_calls = tool_calls

            result.append(msg)

        return result
    

    @classmethod
    def tools(cls, tools: list[AITool]) -> list[Tool]:

        formatted_tools: list[Tool] = []

        for tool in tools:
            formatted_tools.append(Tool(
                type="function",
                function=Tool.Function(
                    name=tool.name,
                    description=tool.description,
                    parameters=Tool.Function.Parameters.model_validate(tool.input_schema),
                ))
            )

        return formatted_tools