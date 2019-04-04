# automatically generated by the FlatBuffers compiler, do not modify

# namespace: ppx

import flatbuffers

class LogNormal(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsLogNormal(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogNormal()
        x.Init(buf, n + offset)
        return x

    # LogNormal
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LogNormal
    def Mean(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # LogNormal
    def Stddev(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def LogNormalStart(builder): builder.StartObject(2)
def LogNormalAddMean(builder, mean): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(mean), 0)
def LogNormalAddStddev(builder, stddev): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(stddev), 0)
def LogNormalEnd(builder): return builder.EndObject()