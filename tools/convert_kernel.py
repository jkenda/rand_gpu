#!/usr/bin/python3

import sys

if len(sys.argv) < 2:
    exit(1)

dst = ""
src = None

with open(sys.argv[1], "r") as f:
    src = f.readlines()

dst = """
#include <string>
static const std::string KERNEL_SOURCE =
"""

for line in src:
    if not line.isspace():
        dst += f'"{line[:-1]}\\n"\n'
dst = f"{dst[:-1]};\n"

with open("kernel.hpp", "w") as f:
    f.write(dst)
