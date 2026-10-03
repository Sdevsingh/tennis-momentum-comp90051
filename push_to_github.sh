#!/usr/bin/env bash
# One-step: create the GitHub repo and invite teammates.
#
# Prerequisite: GitHub CLI installed and logged in as yourself:
#     gh auth login          # (run once, interactive)
# Check with: gh auth status
#
# Then from inside this folder:
#     ./push_to_github.sh
#
# What it does:
#   1. creates a PRIVATE repo  <OWNER>/<REPO>  on your account
#   2. pushes this scaffold to it
#   3. invites your two teammates as collaborators (push access)
#
# Edit the three variables below first.

set -euo pipefail

# --- EDIT THESE ---------------------------------------------------------------
REPO_NAME="tennis-momentum-comp90051"      # the repo name on GitHub
VISIBILITY="private"                        # private | public
# GitHub USERNAMES of your teammates (NOT their Canvas links / emails).
# Ask Benjamin and Faisal for their GitHub usernames and put them here.
TEAMMATES=(
  "BENJAMIN_GITHUB_USERNAME"
  "FAISAL_GITHUB_USERNAME"
)
# ------------------------------------------------------------------------------

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI 'gh' not found. Install from https://cli.github.com/ then run: gh auth login"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: not logged in. Run:  gh auth login   then re-run this script."
  exit 1
fi

OWNER="$(gh api user -q .login)"
echo "Creating ${VISIBILITY} repo ${OWNER}/${REPO_NAME} and pushing..."

# Create the repo from the current folder and push the existing commit.
gh repo create "${REPO_NAME}" --"${VISIBILITY}" --source=. --remote=origin --push

echo "Inviting teammates..."
for u in "${TEAMMATES[@]}"; do
  if [[ "$u" == *GITHUB_USERNAME* ]]; then
    echo "  [skip] placeholder '${u}' — edit TEAMMATES in this script first."
    continue
  fi
  echo "  + ${u}"
  gh api -X PUT "repos/${OWNER}/${REPO_NAME}/collaborators/${u}" -f permission=push >/dev/null \
    && echo "    invited (they'll get an email / GitHub notification to accept)" \
    || echo "    FAILED to invite ${u} — check the username is correct."
done

echo
echo "Done. Repo: https://github.com/${OWNER}/${REPO_NAME}"
echo "Teammates must accept the invite before they can push."
