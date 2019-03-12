# automatically generated by the FlatBuffers compiler, do not modify

# namespace: ppx

import flatbuffers

class Tag(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsTag(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Tag()
        x.Init(buf, n + offset)
        return x

    # Tag
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Tag
    def Address(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Tag
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Tag
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TagStart(builder): builder.StartObject(3)
def TagAddAddress(builder, address): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(address), 0)
def TagAddName(builder, name): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def TagAddValue(builder, value): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(value), 0)
def TagEnd(builder): return builder.EndObject()
