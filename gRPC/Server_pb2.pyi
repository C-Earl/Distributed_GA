from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EMPTY(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class get_genes_REP(_message.Message):
    __slots__ = ["genes"]
    GENES_FIELD_NUMBER: _ClassVar[int]
    genes: bytes
    def __init__(self, genes: _Optional[bytes] = ...) -> None: ...

class get_genes_REQ(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class return_results_REP(_message.Message):
    __slots__ = ["success"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class return_results_REQ(_message.Message):
    __slots__ = ["fitness", "genes", "id"]
    FITNESS_FIELD_NUMBER: _ClassVar[int]
    GENES_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    fitness: float
    genes: bytes
    id: str
    def __init__(self, id: _Optional[str] = ..., genes: _Optional[bytes] = ..., fitness: _Optional[float] = ...) -> None: ...
