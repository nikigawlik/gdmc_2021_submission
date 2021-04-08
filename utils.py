

from typing import MutableSequence


class WrappingList(MutableSequence):
    def __init__(self, l=[]):
        if type(l) is not list:
            raise ValueError()

        self._inner_list = l

    def __len__(self):
        return len(self._inner_list)

    def __delitem__(self, index):
        index = self.__validindex(index)
        self._inner_list.__delitem__(index)

    def insert(self, index, value):
        index = self.__validindex(index)
        self._inner_list.insert(index, value)

    def __setitem__(self, index, value):
        index = self.__validindex(index)
        self._inner_list.__setitem__(index, value)

    def __getitem__(self, index):
        index = self.__validindex(index)
        return self._inner_list.__getitem__(index)

    def __validindex(self, index):
        return index % len(self._inner_list)


lst = [0,2,3,41,2]

wlst = WrappingList(lst)

for i in range(-10, 10):
    print(wlst[i])