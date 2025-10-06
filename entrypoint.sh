#!/bin/sh
if [ "$1" = "--auth" ]; then
  # Run auth command
  exec bun run dist/main.js auth
else
  # Default command
  exec bun run dist/main.js start -r 15 -g "$GH_TOKEN" "$@"
fi

