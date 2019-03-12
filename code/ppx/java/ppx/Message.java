// automatically generated by the FlatBuffers compiler, do not modify

package ppx;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class Message extends Table {
  public static Message getRootAsMessage(ByteBuffer _bb) { return getRootAsMessage(_bb, new Message()); }
  public static Message getRootAsMessage(ByteBuffer _bb, Message obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public static boolean MessageBufferHasIdentifier(ByteBuffer _bb) { return __has_identifier(_bb, "PPXF"); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; vtable_start = bb_pos - bb.getInt(bb_pos); vtable_size = bb.getShort(vtable_start); }
  public Message __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public byte bodyType() { int o = __offset(4); return o != 0 ? bb.get(o + bb_pos) : 0; }
  public Table body(Table obj) { int o = __offset(6); return o != 0 ? __union(obj, o) : null; }

  public static int createMessage(FlatBufferBuilder builder,
      byte body_type,
      int bodyOffset) {
    builder.startObject(2);
    Message.addBody(builder, bodyOffset);
    Message.addBodyType(builder, body_type);
    return Message.endMessage(builder);
  }

  public static void startMessage(FlatBufferBuilder builder) { builder.startObject(2); }
  public static void addBodyType(FlatBufferBuilder builder, byte bodyType) { builder.addByte(0, bodyType, 0); }
  public static void addBody(FlatBufferBuilder builder, int bodyOffset) { builder.addOffset(1, bodyOffset, 0); }
  public static int endMessage(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
  public static void finishMessageBuffer(FlatBufferBuilder builder, int offset) { builder.finish(offset, "PPXF"); }
  public static void finishSizePrefixedMessageBuffer(FlatBufferBuilder builder, int offset) { builder.finishSizePrefixed(offset, "PPXF"); }
}

