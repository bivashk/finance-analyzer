#!/bin/sh
set -eu
export PORT="${PORT:-8001}"
exec python api.py
