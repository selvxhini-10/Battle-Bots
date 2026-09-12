#!/bin/sh
set -eu
cd "$(dirname "$0")"

headless=false
if [ "${1-}" = "--headless" ]; then
    headless=true
    shift
fi

echo "Running the Solemates synthetic demo..."
python3 -m solemates --synthetic --output demo_output "$@"

result_path="$PWD/demo_output/matches.png"
echo "Opening $result_path"
if [ "$headless" = false ] && command -v open >/dev/null 2>&1; then
    open "$result_path"
else
    echo "Headless mode: open the image above manually."
fi
