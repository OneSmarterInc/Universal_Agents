#!/bin/sh
# Download Qwen's own Qwen3-1.7B Q8_0 file at a fixed revision, check it, and write the
# Modelfile that compose's model-init uses to build kormic-local (docs/model-setup.md).
# Run from the project folder: sh deploy/fetch-model.sh
set -eu

REVISION=90862c4b9d2787eaed51d12237eafdfe7c5f6077
FILE=Qwen3-1.7B-Q8_0.gguf
SHA256=061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a
URL="https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/$REVISION/$FILE"

mkdir -p models
if [ ! -f "models/$FILE" ]; then
    curl --fail --location --proto '=https' --tlsv1.2 -o "models/$FILE.part" "$URL"
    mv "models/$FILE.part" "models/$FILE"
fi
echo "$SHA256  models/$FILE" | sha256sum -c - || {
    echo "Checksum mismatch: models/$FILE is not the approved file. Delete it and run again." >&2
    exit 1
}

cat > models/Modelfile <<EOF
FROM ./$FILE
PARAMETER temperature 0
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER num_ctx 16384
PARAMETER repeat_penalty 1
EOF
echo "models/$FILE checked; models/Modelfile written."
