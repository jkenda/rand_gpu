#!/usr/bin/python3

import sys
import os
import re

if len(sys.argv) < 3:
    exit(1)

filenames = [
    "kiss09.cl",
    "lcg12864.cl",
    "lfib.cl",
    "mrg63k3a.cl",
    "msws.cl",
    "mt19937.cl",
    "mwc64x.cl",
    "pcg6432.cl",
    "philox2x32_10.cl",
    "ran2.cl",
    "tinymt64.cl",
    "tyche.cl",
    "tyche_i.cl",
    "well512.cl",
    "xorshift6432star.cl",
]

multiline_comment = False
prev_line_was_a_preprocessor_directive = False

directory = sys.argv[1]

dst = """
#pragma once
#include <string>
static const std::string KERNEL_SOURCE[] = {
"""

for filename in filenames:
    with open(os.path.join(directory, filename), "r") as f:
        lines = f.readlines()

        for line in lines:
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

            dst += f'\t"{line}"\n'

        dst = dst[:-1] + ",\n"

dst = dst.replace("__global", "global")
dst = dst.replace("__kernel", "kernel")
dst = re.sub(" +", " ", dst)
dst += "};\n"

with open(sys.argv[2], "w") as f:
    f.write(dst)
