#!/usr/bin/env python3
"""IDVerify entry point.

    python main.py                 # desktop interface
    python main.py scan --card card.jpg
    python main.py enroll --card card.jpg --face selfie.jpg
    python main.py verify --card card.jpg --face selfie.jpg
"""

import sys

from idverify.cli import main

if __name__ == "__main__":
    sys.exit(main())