<?php
// automatically generated by the FlatBuffers compiler, do not modify

namespace ppx;

use \Google\FlatBuffers\Struct;
use \Google\FlatBuffers\Table;
use \Google\FlatBuffers\ByteBuffer;
use \Google\FlatBuffers\FlatBufferBuilder;

class Observe extends Table
{
    /**
     * @param ByteBuffer $bb
     * @return Observe
     */
    public static function getRootAsObserve(ByteBuffer $bb)
    {
        $obj = new Observe();
        return ($obj->init($bb->getInt($bb->getPosition()) + $bb->getPosition(), $bb));
    }

    public static function ObserveIdentifier()
    {
        return "PPXF";
    }

    public static function ObserveBufferHasIdentifier(ByteBuffer $buf)
    {
        return self::__has_identifier($buf, self::ObserveIdentifier());
    }

    /**
     * @param int $_i offset
     * @param ByteBuffer $_bb
     * @return Observe
     **/
    public function init($_i, ByteBuffer $_bb)
    {
        $this->bb_pos = $_i;
        $this->bb = $_bb;
        return $this;
    }

    public function getAddress()
    {
        $o = $this->__offset(4);
        return $o != 0 ? $this->__string($o + $this->bb_pos) : null;
    }

    public function getName()
    {
        $o = $this->__offset(6);
        return $o != 0 ? $this->__string($o + $this->bb_pos) : null;
    }

    /**
     * @return byte
     */
    public function getDistributionType()
    {
        $o = $this->__offset(8);
        return $o != 0 ? $this->bb->getByte($o + $this->bb_pos) : \ppx\Distribution::NONE;
    }

    /**
     * @returnint
     */
    public function getDistribution($obj)
    {
        $o = $this->__offset(10);
        return $o != 0 ? $this->__union($obj, $o) : null;
    }

    public function getValue()
    {
        $obj = new Tensor();
        $o = $this->__offset(12);
        return $o != 0 ? $obj->init($this->__indirect($o + $this->bb_pos), $this->bb) : 0;
    }

    /**
     * @param FlatBufferBuilder $builder
     * @return void
     */
    public static function startObserve(FlatBufferBuilder $builder)
    {
        $builder->StartObject(5);
    }

    /**
     * @param FlatBufferBuilder $builder
     * @return Observe
     */
    public static function createObserve(FlatBufferBuilder $builder, $address, $name, $distribution_type, $distribution, $value)
    {
        $builder->startObject(5);
        self::addAddress($builder, $address);
        self::addName($builder, $name);
        self::addDistributionType($builder, $distribution_type);
        self::addDistribution($builder, $distribution);
        self::addValue($builder, $value);
        $o = $builder->endObject();
        return $o;
    }

    /**
     * @param FlatBufferBuilder $builder
     * @param StringOffset
     * @return void
     */
    public static function addAddress(FlatBufferBuilder $builder, $address)
    {
        $builder->addOffsetX(0, $address, 0);
    }

    /**
     * @param FlatBufferBuilder $builder
     * @param StringOffset
     * @return void
     */
    public static function addName(FlatBufferBuilder $builder, $name)
    {
        $builder->addOffsetX(1, $name, 0);
    }

    /**
     * @param FlatBufferBuilder $builder
     * @param byte
     * @return void
     */
    public static function addDistributionType(FlatBufferBuilder $builder, $distributionType)
    {
        $builder->addByteX(2, $distributionType, 0);
    }

    public static function addDistribution(FlatBufferBuilder $builder, $offset)
    {
        $builder->addOffsetX(3, $offset, 0);
    }

    /**
     * @param FlatBufferBuilder $builder
     * @param int
     * @return void
     */
    public static function addValue(FlatBufferBuilder $builder, $value)
    {
        $builder->addOffsetX(4, $value, 0);
    }

    /**
     * @param FlatBufferBuilder $builder
     * @return int table offset
     */
    public static function endObserve(FlatBufferBuilder $builder)
    {
        $o = $builder->endObject();
        return $o;
    }
}