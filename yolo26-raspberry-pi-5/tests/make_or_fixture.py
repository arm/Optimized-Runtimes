"""Create a zero-weight, three-class native fixture for ABI tests, not accuracy."""

import math
import re
import struct
import sys
from pathlib import Path


def main() -> None:
    runtime = Path(__file__).resolve().parents[1] / "src/runtime"
    slots = {
        name: int(index)
        for name, index in re.findall(
            r"(k\w+) = (\d+)", (runtime / "fixed_weight_slots.inc").read_text()
        )
    }
    sizes: dict[int, int] = {}
    for source in runtime.glob("fixed_graph*.inc"):
        for dtype, shape, name in re.findall(
            r"graph\.DefineWeight\(\s*ValueType::(\w+),\s*\{([^}]+)\},\s*WeightSlot::(\w+)\)",
            source.read_text(),
        ):
            shape = shape.replace("ClassificationWidth()", "64").replace("ClassCount()", "3")
            shape = re.sub(r"Constant\((\d+)\)", r"\1", shape)
            elements = math.prod(int(extent.strip()) for extent in shape.split(","))
            size = elements * {"kFp32": 4, "kFp16": 2}[dtype]
            index = slots[name]
            if index in sizes and sizes[index] != size:
                raise ValueError(f"Conflicting shapes for {name}")
            sizes[index] = size
    if set(sizes) != set(range(208)):
        raise ValueError("Fixture must cover all 208 runtime weight slots")
    data = bytearray(64 + 208 * 16)
    struct.pack_into("<8sIIIIQ", data, 0, b"Y26MOD01", 1, 320, 3, 208, len(data))
    for index in range(208):
        data.extend(bytes(-len(data) % 64))
        struct.pack_into("<QQ", data, 64 + index * 16, len(data), sizes[index])
        data.extend(bytes(sizes[index]))
    Path(sys.argv[1]).write_bytes(data)


if __name__ == "__main__":
    main()
