"""Phonemize text to token IDs. Called by the Rust server via subprocess.

Usage: python src/scripts/phonemize.py "Hello world"
Output: JSON array of int token IDs on stdout.
"""

import json
import os
import sys

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak.dylib")

from smalltts.data.phonemization.phonemes import get_token_ids  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/scripts/phonemize.py <text>", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(get_token_ids(sys.argv[1])))
