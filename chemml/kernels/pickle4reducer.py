from multiprocessing.reduction import ForkingPickler, AbstractReducer


class ForkingPickler4(ForkingPickler):
    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)


def dumps(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dumps(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dumps = dumps
