"""Verify the SHA-256 checksums of the published files.

The raw call logs runs/final/raw_*.jsonl are append-only: the run of 23.10.2026, the complementary
experiments (v14) and the clean-data re-check (v15) appended to the same files. Each checksum file
records the state at one stage; a recorded hash must therefore match either the whole file or one of
its line prefixes. This script reports which.

python analysis/verify_checksums.py
"""
import glob
import hashlib
import os

RUN = os.path.join("runs", "final")


def prefix_match(path, target):
    """Return the number of leading lines whose SHA-256 equals target, or None."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            h.update(line)
            n += 1
            if h.copy().hexdigest() == target:
                return n
    return None


def main():
    ok = True
    for ck in sorted(glob.glob(os.path.join(RUN, "checksums_*.txt"))):
        print(f"== {os.path.basename(ck)}")
        for line in open(ck, encoding="utf-8-sig"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            target, name = parts[0], parts[1]
            path = os.path.join(RUN, name)
            if not os.path.exists(path):
                print(f"   ABSENT   {name}")
                ok = False
                continue
            whole = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if whole == target:
                print(f"   OK       {name} (fichier entier)")
                continue
            n = prefix_match(path, target) if name.endswith(".jsonl") else None
            if n:
                print(f"   OK       {name} ({n} premières lignes)")
            else:
                print(f"   ÉCHEC    {name}")
                ok = False
    print("\nTout est conforme." if ok else "\nAu moins un fichier ne correspond pas.")


if __name__ == "__main__":
    main()
