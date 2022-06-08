#!/usr/bin/python3

import sys

if len(sys.argv) < 3:
    exit(1)

dst = ""
src = None
multiline_comment = False
prev_line_was_a_preprocessor_directive = False

with open(sys.argv[1], "r") as f:
    src = f.readlines()

dst = 'static const char *KERNEL_SOURCE = \n'

for line in src:
    if multiline_comment:
        comment_end = line.find("*/")
        if comment_end == -1: continue
        else:
            line = line[comment_end+2:]
            multiline_comment = False

    comment_start = line.find("//")
    if comment_start != -1:
        line = line[:comment_start]

    comment_start = line.find("/*")
    if comment_start != -1:
        line = line[:comment_start]
        multiline_comment = True

    line = line.strip()

    if line and not line.isspace():
        dst += '"'
        if line.startswith("#") or prev_line_was_a_preprocessor_directive:
            dst += '\\n'
        dst += line + '"\n'

        if line.startswith("#"):
            prev_line_was_a_preprocessor_directive = True
        else:
            prev_line_was_a_preprocessor_directive = False

dst = f"{dst[:-1]};\n"

with open(sys.argv[2], "w") as f:
    f.write(dst)
