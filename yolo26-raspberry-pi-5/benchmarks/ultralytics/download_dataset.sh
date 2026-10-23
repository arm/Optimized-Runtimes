#!/usr/bin/env bash
set -euo pipefail

revision="$1"
output="$2"
base="https://huggingface.co/datasets/Arm/coco-128-by-sa/resolve/$revision"
manifest="LICENSE_MANIFEST.csv"

# The manifest is the dataset's lightweight index of image paths. Publish it
# only after curl succeeds so an interrupted transfer cannot become the index.
mkdir -p "$output"
curl --fail --location --retry 3 --silent --show-error \
  --output "$output/$manifest.part" "$base/$manifest"
mv "$output/$manifest.part" "$output/$manifest"

paths=(
  data.yaml
)

# The first three CSV fields contain no commas. Validate them before using
# repository-controlled values as local paths and download URLs.
while IFS=, read -r image_id file_name relative_path _; do
  if [[ "$image_id" == image_id ]]; then
    continue
  fi
  [[ "$image_id" =~ ^[0-9]+$ ]]
  [[ "$file_name" =~ ^[0-9]{12}\.jpg$ ]]
  [[ "$relative_path" == "images/train2017/$file_name" ]]
  paths+=("$relative_path" "labels/train2017/${file_name%.jpg}.txt")
done < "$output/$manifest"

# Build one bounded parallel curl operation. --next separates the options for
# each transfer, and every transfer writes to a temporary path.
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

# Publish the dataset files only after every parallel transfer has succeeded.
for path in "${paths[@]}"; do
  mv "$output/$path.part" "$output/$path"
done

# Ultralytics resolves "path: ." against its working directory rather than the
# source YAML's directory. Generate an absolute-path benchmark configuration,
# escaping single quotes according to YAML rules. Written last, this file is
# also the Make target that records successful dataset preparation.
escaped_output=${output//\'/\'\'}
{
  printf "path: '%s'\n" "$escaped_output"
  tail -n +2 "$output/data.yaml"
} > "$output/benchmark.yaml.part"
mv "$output/benchmark.yaml.part" "$output/benchmark.yaml"
