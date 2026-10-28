#!/usr/bin/env bash
# Quick manual chat helper:  scripts/smoke.sh C1009 s1 "Where is order SS-10512?"
curl -s -X POST http://localhost:8000/chat -H "X-Customer-Id: $1" -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"session_id":sys.argv[1],"message":sys.argv[2]}))' "$2" "$3")"; echo
