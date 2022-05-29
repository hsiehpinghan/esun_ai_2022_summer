import abc

class AbstractValueObject(abc.ABC):

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError("__hash__ not implement !!!")

    @abc.abstractmethod
    def __eq__(self, other: object):
        raise NotImplementedError("__eq__ not implement !!!")
