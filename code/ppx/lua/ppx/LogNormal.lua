-- automatically generated by the FlatBuffers compiler, do not modify

-- namespace: ppx

local flatbuffers = require('flatbuffers')

local LogNormal = {} -- the module
local LogNormal_mt = {} -- the class metatable

function LogNormal.New()
    local o = {}
    setmetatable(o, {__index = LogNormal_mt})
    return o
end
function LogNormal.GetRootAsLogNormal(buf, offset)
    local n = flatbuffers.N.UOffsetT:Unpack(buf, offset)
    local o = LogNormal.New()
    o:Init(buf, n + offset)
    return o
end
function LogNormal_mt:Init(buf, pos)
    self.view = flatbuffers.view.New(buf, pos)
end
function LogNormal_mt:Mean()
    local o = self.view:Offset(4)
    if o ~= 0 then
        local x = self.view:Indirect(o + self.view.pos)
        local obj = require('ppx.Tensor').New()
        obj:Init(self.view.bytes, x)
        return obj
    end
end
function LogNormal_mt:Stddev()
    local o = self.view:Offset(6)
    if o ~= 0 then
        local x = self.view:Indirect(o + self.view.pos)
        local obj = require('ppx.Tensor').New()
        obj:Init(self.view.bytes, x)
        return obj
    end
end
function LogNormal.Start(builder) builder:StartObject(2) end
function LogNormal.AddMean(builder, mean) builder:PrependUOffsetTRelativeSlot(0, mean, 0) end
function LogNormal.AddStddev(builder, stddev) builder:PrependUOffsetTRelativeSlot(1, stddev, 0) end
function LogNormal.End(builder) return builder:EndObject() end

return LogNormal -- return the module