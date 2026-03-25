#!/usr/bin/env bash

set -euo pipefail

json_dir="${JSON_DIR:-$HOME/code/sand-atlas/_data/json}"
sands_dir="${SANDS_DIR:-/Volumes/PRJ-SciGEMData/sand-atlas/sands}"
website_sands_dir="${WEBSITE_SANDS_DIR:-$HOME/code/sand-atlas/assets/sands}"

if [[ ! -d "$json_dir" ]]; then
	echo "JSON directory not found: $json_dir" >&2
	exit 1
fi

if [[ ! -d "$sands_dir" ]]; then
	echo "Sands directory not found: $sands_dir" >&2
	exit 1
fi

if [[ ! -d "$website_sands_dir" ]]; then
	echo "Website sands directory not found: $website_sands_dir" >&2
	exit 1
fi

found_match=0

for json_file in "$json_dir"/*.json; do
	if [[ ! -e "$json_file" ]]; then
		continue
	fi

	sample_name=$(basename "$json_file" .json)
	label_file="$sands_dir/$sample_name/labelled.tif"

	if [[ ! -f "$label_file" ]]; then
		if [[ -d "$sands_dir/$sample_name" ]]; then
			if [[ -z "$(find "$sands_dir/$sample_name" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
				echo "Skipping $sample_name: sample folder exists but is empty ($sands_dir/$sample_name)" >&2
			else
				echo "Skipping $sample_name: missing labelled TIFF at $label_file" >&2
			fi
		else
			echo "Skipping $sample_name: sample folder does not exist ($sands_dir/$sample_name)" >&2
		fi
		continue
	fi

	output_file="$sands_dir/$sample_name/summary.csv"
	website_output_file="$website_sands_dir/$sample_name.csv"

	echo "Processing $sample_name"
	sand_atlas_properties "$json_file" "$label_file" --output "$output_file"
	cp "$output_file" "$website_output_file"
	found_match=1
done

for label_file in "$sands_dir"/*/labelled.tif; do
	if [[ ! -e "$label_file" ]]; then
		continue
	fi

	sample_dir=$(dirname "$label_file")
	sample_name=$(basename "$sample_dir")
	json_file="$json_dir/$sample_name.json"

	if [[ ! -f "$json_file" ]]; then
		echo "No JSON found for $sample_name: expected $json_file" >&2
	fi
done

if [[ "$found_match" -eq 0 ]]; then
	echo "No matching JSON/labelled TIFF pairs found between $json_dir and $sands_dir" >&2
	exit 1
fi