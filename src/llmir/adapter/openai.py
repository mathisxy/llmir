from typing import cast
import base64
import json

from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionMessageToolCallParam,
)
    
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.shared_params.function_definition import FunctionDefinition

from ..messages import AIMessages, AIMessageToolResponse, AIRoles
from ..chunks import AIChunkText, AIChunkImageURL, AIChunkFile, AIChunkToolCall
from ..tools import AITool
from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter):

    decode_text_file_types: set[str] = {"text/plain", "application/json"}
    set_file_origin_to_user: set[AIRoles] = {AIRoles.TOOL, AIRoles.MODEL}
    unsupported_file_type_message: str | None = "[Unsupported file type: {mimetype}, size: {length} bytes, name: {filename}]"

    @classmethod
    def chat(cls, messages: list[AIMessages]) -> list[ChatCompletionMessageParam]:
        
        result: list[ChatCompletionMessageParam] = []

        for message in messages:

            match message.role.value:

                case AIRoles.SYSTEM.value:

                    formatted_text_chunks: list[ChatCompletionContentPartTextParam] = []
                    for chunk in message.chunks:
                        match chunk:
                            case AIChunkText():
                                formatted_text_chunks.append(cls.text_chunk(chunk))
                            case AIChunkFile():
                                if chunk.mimetype in cls.decode_text_file_types:
                                    formatted_text_chunks.append(cls.text_file_chunk(chunk))
                                elif message.role in cls.set_file_origin_to_user:
                                    result.append(ChatCompletionUserMessageParam(
                                        role=AIRoles.USER.value,
                                        content=[cls.content_chunk(chunk)]
                                    ))
                                else:
                                    raise ValueError(f"Unsupported file type in system message: {chunk.mimetype}")
                            case _:
                                raise ValueError(f"Unsupported chunk type in system message: {type(chunk)}")
                            
                    result.append(ChatCompletionSystemMessageParam(
                        role=message.role.value,
                        content=formatted_text_chunks,
                    ))
                            
                case AIRoles.USER.value:
                    
                    formatted_content_chunks: list[ChatCompletionContentPartParam] = []
                    for chunk in message.chunks:
                        match chunk:
                            case AIChunkText() | AIChunkImageURL() | AIChunkFile():
                                formatted_content_chunks.append(cls.content_chunk(chunk))
                            case _:
                                raise ValueError(f"Unsupported chunk type in user message: {type(chunk)}")
                            
                    result.append(ChatCompletionUserMessageParam(
                        role=message.role.value,
                        content=formatted_content_chunks,
                    ))
                            
                case AIRoles.MODEL.value:

                    formatted_text_chunks: list[ChatCompletionContentPartTextParam] = []
                    formatted_tool_calls: list[ChatCompletionMessageToolCallParam] = []
                    for chunk in message.chunks:
                        match chunk:
                            case AIChunkText():
                                formatted_text_chunks.append(cls.text_chunk(chunk))
                            case AIChunkToolCall():
                                formatted_tool_calls.append(cls.tool_call_chunk(chunk))
                            case AIChunkFile():
                                if chunk.mimetype in cls.decode_text_file_types:
                                    formatted_text_chunks.append(cls.text_file_chunk(chunk))
                                elif message.role in cls.set_file_origin_to_user:
                                    result.append(ChatCompletionUserMessageParam(
                                        role=AIRoles.USER.value,
                                        content=[cls.content_chunk(chunk)]
                                    ))
                                else:
                                    raise ValueError(f"Unsupported file type in model message: {chunk.mimetype}")
                            case _:
                                raise ValueError(f"Unsupported chunk type in model message: {type(chunk)}")
                            
                    result.append(ChatCompletionAssistantMessageParam(
                        role=message.role.value,
                        content=formatted_text_chunks,
                        tool_calls=formatted_tool_calls,
                    ))
                            
                case AIRoles.TOOL.value:

                    assert isinstance(message, AIMessageToolResponse), f"Tool message must be of type AIMessageToolResponse, but got {type(message)}"

                    text = ""
                    for chunk in message.chunks:
                        match chunk:
                            case AIChunkText():
                                text += chunk.text + "\n"
                            case AIChunkFile():
                                if chunk.mimetype in cls.decode_text_file_types:
                                    text += cls.text_file_chunk(chunk)["text"] + "\n"
                                elif message.role in cls.set_file_origin_to_user:
                                    result.append(ChatCompletionUserMessageParam(
                                        role=AIRoles.USER.value,
                                        content=[cls.content_chunk(chunk)]
                                    ))
                                else:
                                    raise ValueError(f"Unsupported file type in tool message: {chunk.mimetype}")
                            case _:
                                raise ValueError(f"Unsupported chunk type in tool message: {type(chunk)}")
                            
                    result.append(ChatCompletionToolMessageParam(
                        role=message.role.value,
                        content=text,
                        tool_call_id=message.id,
                    ))
                    

        return result


    @classmethod
    def content_chunk(cls, chunk: AIChunkText | AIChunkFile | AIChunkImageURL) -> ChatCompletionContentPartParam:

        match chunk:
            case AIChunkText():
                return cls.text_chunk(chunk)
            case AIChunkImageURL():
                return cls.image_url_chunk(chunk)
            case AIChunkFile():
                if chunk.mimetype.startswith("image/"):
                    return cls.image_file_chunk(chunk)
                elif chunk.mimetype in cls.decode_text_file_types:
                    return cls.text_file_chunk(chunk)
                elif cls.unsupported_file_type_message is not None:
                    return ChatCompletionContentPartTextParam(  # Fallback: represent as text
                        type="text",
                        text=cls.unsupported_file_type_message.format(
                            mimetype=chunk.mimetype,
                            length=len(chunk.bytes),
                            name=chunk.name
                        )
                    )
                raise ValueError(f"Unsupported file type for OpenAI: {chunk.mimetype}")
            case _:
                raise ValueError(f"Unsupported chunk type: {type(chunk)}")
            
    @classmethod
    def text_chunk(cls, chunk: AIChunkText) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(
            type="text",
            text=chunk.text,
        )
    
    @classmethod
    def image_url_chunk(cls, chunk: AIChunkImageURL) -> ChatCompletionContentPartImageParam:
        return ChatCompletionContentPartImageParam(
            type="image_url",
            image_url=ImageURL(
                url=chunk.url
            )
        )
    
    @classmethod
    def text_file_chunk(cls, chunk: AIChunkFile) -> ChatCompletionContentPartTextParam:
        if chunk.mimetype.startswith("text/"):
            text = chunk.bytes.decode("utf-8")
            return ChatCompletionContentPartTextParam(
                type="text",
                text=text
            )
        else:
            raise ValueError(f"Not a text file: {chunk.mimetype}")
        
    @classmethod
    def image_file_chunk(cls, chunk: AIChunkFile) -> ChatCompletionContentPartImageParam:
        if chunk.mimetype.startswith("image/"):
            base64_image = base64.b64encode(chunk.bytes).decode("utf-8")
            return ChatCompletionContentPartImageParam(
                type="image_url",
                image_url=ImageURL(
                    url=f"data:{chunk.mimetype};base64,{base64_image}"
                )
            )
        else:
            raise ValueError(f"Not an image file: {chunk.mimetype}")

   
    @classmethod
    def tool_call_chunk(cls, chunk: AIChunkToolCall) -> ChatCompletionMessageToolCallParam:

        return ChatCompletionMessageToolCallParam(
                    id=chunk.id,
                    type="function",
                    function=Function(
                        name=chunk.name,
                        arguments=json.dumps(chunk.arguments)
                    )
                )
    
    @classmethod
    def tools(cls, tools: list[AITool]) -> list[ChatCompletionFunctionToolParam]:

        formatted_tools: list[ChatCompletionFunctionToolParam] = []

        for tool in tools:
            formatted_tools.append(ChatCompletionFunctionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=cast(dict[str, object], tool.input_schema),
                )
            ))

        return formatted_tools