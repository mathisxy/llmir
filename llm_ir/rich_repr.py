from pydantic import BaseModel

class RichReprMixin(BaseModel):

    MAX_CHARS_PER_VALUE: int = 2000

    def __rich_repr__(self):

        for name in self.__class__.model_fields:
            value = getattr(self, name)
            length = len(str(value))

            if length > self.MAX_CHARS_PER_VALUE:
                yield name, f"<object of length: {len(str(value))}>"
            else:
                yield name, value