#!/usr/bin/env bash

CSV_FILE="test.csv"

tail -n +2 "$CSV_FILE" | while IFS=',' read -r lab hpc; do
    echo "Running on: $lab"


done