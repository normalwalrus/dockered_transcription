#!/usr/bin/env bash
# Downloads the CTranslate2 conversion of whisper-large-v3-turbo into ./models.
# Usage: ./download_model.sh [hf_repo_id] [target_dir]
#
# Uses huggingface_hub from the host Python if available, otherwise runs the
# download inside the whisper-turbo-api Docker image (built via docker compose build).
set -euo pipefail

REPO="${1:-deepdml/faster-whisper-large-v3-turbo-ct2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${2:-$HERE/models/whisper-large-v3-turbo-ct2}"
IMAGE="whisper-turbo-api:latest"

PY_SNIPPET='
import os, sys
from huggingface_hub import snapshot_download
repo, target = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=target)
print("ok")
'

mkdir -p "$TARGET"
echo "Downloading $REPO -> $TARGET"

if python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -c "$PY_SNIPPET" "$REPO" "$TARGET"
elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "(host has no huggingface_hub; downloading inside $IMAGE)"
    docker run --rm -u "$(id -u):$(id -g)" \
        -e HF_HOME=/tmp/hf -v "$TARGET:/dl" "$IMAGE" \
        python3 -c "$PY_SNIPPET" "$REPO" /dl
else
    echo "Need either 'pip install huggingface_hub' on the host or a built image ($IMAGE)."
    echo "Run: docker compose build   and re-run this script."
    exit 1
fi

echo
echo "Done. Contents:"
ls -lh "$TARGET"
