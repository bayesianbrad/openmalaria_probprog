// Code generated by the FlatBuffers compiler. DO NOT EDIT.

package ppx

import (
	flatbuffers "github.com/google/flatbuffers/go"
)

type Observe struct {
	_tab flatbuffers.Table
}

func GetRootAsObserve(buf []byte, offset flatbuffers.UOffsetT) *Observe {
	n := flatbuffers.GetUOffsetT(buf[offset:])
	x := &Observe{}
	x.Init(buf, n+offset)
	return x
}

func (rcv *Observe) Init(buf []byte, i flatbuffers.UOffsetT) {
	rcv._tab.Bytes = buf
	rcv._tab.Pos = i
}

func (rcv *Observe) Table() flatbuffers.Table {
	return rcv._tab
}

func (rcv *Observe) Address() []byte {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(4))
	if o != 0 {
		return rcv._tab.ByteVector(o + rcv._tab.Pos)
	}
	return nil
}

func (rcv *Observe) Name() []byte {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(6))
	if o != 0 {
		return rcv._tab.ByteVector(o + rcv._tab.Pos)
	}
	return nil
}

func (rcv *Observe) DistributionType() byte {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(8))
	if o != 0 {
		return rcv._tab.GetByte(o + rcv._tab.Pos)
	}
	return 0
}

func (rcv *Observe) MutateDistributionType(n byte) bool {
	return rcv._tab.MutateByteSlot(8, n)
}

func (rcv *Observe) Distribution(obj *flatbuffers.Table) bool {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(10))
	if o != 0 {
		rcv._tab.Union(obj, o)
		return true
	}
	return false
}

func (rcv *Observe) Value(obj *Tensor) *Tensor {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(12))
	if o != 0 {
		x := rcv._tab.Indirect(o + rcv._tab.Pos)
		if obj == nil {
			obj = new(Tensor)
		}
		obj.Init(rcv._tab.Bytes, x)
		return obj
	}
	return nil
}

func ObserveStart(builder *flatbuffers.Builder) {
	builder.StartObject(5)
}
func ObserveAddAddress(builder *flatbuffers.Builder, address flatbuffers.UOffsetT) {
	builder.PrependUOffsetTSlot(0, flatbuffers.UOffsetT(address), 0)
}
func ObserveAddName(builder *flatbuffers.Builder, name flatbuffers.UOffsetT) {
	builder.PrependUOffsetTSlot(1, flatbuffers.UOffsetT(name), 0)
}
func ObserveAddDistributionType(builder *flatbuffers.Builder, distributionType byte) {
	builder.PrependByteSlot(2, distributionType, 0)
}
func ObserveAddDistribution(builder *flatbuffers.Builder, distribution flatbuffers.UOffsetT) {
	builder.PrependUOffsetTSlot(3, flatbuffers.UOffsetT(distribution), 0)
}
func ObserveAddValue(builder *flatbuffers.Builder, value flatbuffers.UOffsetT) {
	builder.PrependUOffsetTSlot(4, flatbuffers.UOffsetT(value), 0)
}
func ObserveEnd(builder *flatbuffers.Builder) flatbuffers.UOffsetT {
	return builder.EndObject()
}
