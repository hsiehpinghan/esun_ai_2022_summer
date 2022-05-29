import abc
import typing

from esun.domain.abstract_value_object import AbstractValueObject

class AbstractId(AbstractValueObject):

    @abc.abstractmethod
    def __init__(self, value: str) -> None:
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        if self is other:
            return True
        if type(self) != type(other):
            return False
        return self.value == other.value
