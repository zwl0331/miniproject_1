#!/bin/zsh
cd /private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/9f6b6eec-676c-4338-a75c-d6002d6d0d06/scratchpad/agentC_mechanics
while ! grep -q 'Q1 DONE' q1.log; do sleep 10; done
PYTHONDONTWRITEBYTECODE=1 /Users/zwl0331/Desktop/Mini_project_1/.venv/bin/python q2.py > q2.log 2>&1
