#!/bin/bash
# Vol Surface App — Quick Start
set -e

cd "$(dirname "$0")"

echo ""
echo "  📈  Vol Surface App"
echo "  ──────────────────"

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
  echo "  → Creating virtual environment…"
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "  → Installing dependencies…"
pip install -q -r requirements.txt

echo ""
echo "  ✓  Starting server at http://localhost:8080"
echo "  ✓  Press Ctrl+C to stop"
echo ""

python app.py
