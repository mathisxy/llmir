from enum import StrEnum


class AIRoles(StrEnum):
    """
    Roles for the LLM messages.

    MODEL has the value assistant, because it is legacy standard to name the role of the model assistant.
    """

    USER = "user"
    MODEL = "assistant"
    SYSTEM = "system"
    TOOL = "tool"