"""IDVerify entry point.

    python main.py                 # desktop interface
    python main.py gui             # desktop interface (same as above)
    python main.py scan --card card.jpg
    python main.py enroll --card card.jpg --face selfie.jpg
    python main.py verify --card card.jpg --face selfie.jpg
    python main.py verify --card card.jpg --camera 0
    python main.py history
    python main.py list
    python main.py forget --id <ID_NUMBER>
    python main.py attend           # live camera attendance / enrollment
    python main.py attend --enroll
    python main.py attend --list
    python main.py attend --export register.csv
    python main.py attend --clear
    python main.py attend --camera 1
"""

from __future__ import annotations

import sys

from idverify.cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
