<?php
// automatically generated by the FlatBuffers compiler, do not modify

namespace ppx;

class MessageBody
{
    const NONE = 0;
    const Handshake = 1;
    const HandshakeResult = 2;
    const Run = 3;
    const RunResult = 4;
    const Sample = 5;
    const SampleResult = 6;
    const Observe = 7;
    const ObserveResult = 8;
    const Tag = 9;
    const TagResult = 10;
    const Reset = 11;

    private static $names = array(
        MessageBody::NONE=>"NONE",
        MessageBody::Handshake=>"Handshake",
        MessageBody::HandshakeResult=>"HandshakeResult",
        MessageBody::Run=>"Run",
        MessageBody::RunResult=>"RunResult",
        MessageBody::Sample=>"Sample",
        MessageBody::SampleResult=>"SampleResult",
        MessageBody::Observe=>"Observe",
        MessageBody::ObserveResult=>"ObserveResult",
        MessageBody::Tag=>"Tag",
        MessageBody::TagResult=>"TagResult",
        MessageBody::Reset=>"Reset",
    );

    public static function Name($e)
    {
        if (!isset(self::$names[$e])) {
            throw new \Exception();
        }
        return self::$names[$e];
    }
}