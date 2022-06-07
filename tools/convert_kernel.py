#!/usr/bin/python3

import sys

if len(sys.argv) < 2:
    exit(1)

dst = ""
src = None
multiline_comment = False

with open(sys.argv[1], "r") as f:
    src = f.readlines()

dst = "static const char *KERNEL_SOURCE =\n"

for line in src:

    if multiline_comment:
        comment_end = line.find("*/")
        if comment_end == -1: continue
        else:
            line = line[comment_end+2:]
            multiline_comment = False

    comment_start = line.find("//")
    line = line[:comment_start]

    comment_start = line.find("/*")
    if comment_start != -1:
        line = line[:comment_start]
        multiline_comment = True

    line = line.strip()

    if line and not line.isspace():
        dst += f'"{line}\\n"\n'

dst = f"{dst};\n"

with open("kernel.hpp", "w") as f:
    f.write(dst)
