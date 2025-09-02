#!/usr/bin/env python

import os
import sys

# prefer local vendor copy of rsl_rl if present
vendor_path = "/home/yoda/Projects/personal_tutorial_isaaclab/test/vendor"
if os.path.isdir(os.path.join(vendor_path, "rsl_rl")) and vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

import rsl_rl  # noqa: E402

print(rsl_rl.__file__)
print(sys.path[0])



