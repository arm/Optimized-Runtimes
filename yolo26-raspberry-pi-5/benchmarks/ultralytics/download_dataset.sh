#!/usr/bin/env bash
set -euo pipefail

revision="$1"
output="$2"
base="https://huggingface.co/datasets/Arm/coco-128-by-sa/resolve/$revision"
manifest="LICENSE_MANIFEST.csv"

mkdir -p "$output"
curl --fail --location --retry 3 --silent --show-error \
  --output "$output/$manifest.part" "$base/$manifest"
mv "$output/$manifest.part" "$output/$manifest"

paths=(
  data.yaml
)

while IFS=, read -r image_id file_name relative_path _; do
  if [[ "$image_id" == image_id ]]; then
    continue
  fi
  [[ "$image_id" =~ ^[0-9]+$ ]]
  [[ "$file_name" =~ ^[0-9]{12}\.jpg$ ]]
  [[ "$relative_path" == "images/train2017/$file_name" ]]
  paths+=("$relative_path" "labels/train2017/${file_name%.jpg}.txt")
done < "$output/$manifest"

arguments=(--parallel --parallel-max 8)
first=true
for path in "${paths[@]}"; do
  mkdir -p "$(dirname "$output/$path")"
  if ! $first; then
    arguments+=(--next)
  fi
  first=false
  arguments+=(
    --fail --location --retry 3 --silent --show-error
    --url "$base/$path" --output "$output/$path.part"
  )
done
curl "${arguments[@]}"

for path in "${paths[@]}"; do
  mv "$output/$path.part" "$output/$path"
done
