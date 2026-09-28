from typing import Union,Literal


class Node:
    def __init__(self,
                 kind:Literal["fld","op","param"],
                 value:Union[str,int,float],
                 args:Union[list,None]=None):
        self.kind = kind
        self.value = value
        self.args = [] if args is None else args

    def __repr__(self):
        return (
            "Node("
            f"kind={self.kind!r},"
            f"value={self.value!r},"
            f"args={self.args!r}"
            ")"
        )
