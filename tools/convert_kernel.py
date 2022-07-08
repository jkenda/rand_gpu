#!/usr/bin/python3

import sys
import os

if len(sys.argv) < 3:
    exit(1)

src = []
multiline_comment = False
prev_line_was_a_preprocessor_directive = False

directory = sys.argv[1]

for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), "r") as f:
        src = src + f.readlines()
        src.append("\n")

dst = "static const char *KERNEL_SOURCE = \n"

for line in src:
    line = line.strip()

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
    while comment_start != -1:
        multiline_comment = True

        comment_end = line.find("*/")
        if comment_end != -1:
            multiline_comment = False
            line = line[:comment_start] + " " + line[comment_end+2:]
        else:
            line = line[:comment_start]
    
        comment_start = line.find("/*")

    if not line or line.isspace():
        continue
    if line.endswith("\\"):
        line = line[:-1]
    else:
        line += "\\n"

    dst += f'"{line}"\n'

dst += ";\n"

with open(sys.argv[2], "w") as f:
    f.write(dst)
