import abc

from esun.domain.abstract_id import AbstractId
from esun.domain.abstract_value_object import AbstractValueObject

class AbstractEntity(AbstractValueObject):

    @abc.abstractmethod
    def __init__(self, id: AbstractId) -> None:
        self._id = id

    @property
    def id(self) -> AbstractId:
        return self._id

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        if self is other:
            return True
        if type(self) != type(other):
            return False
        return self.id == other.id
