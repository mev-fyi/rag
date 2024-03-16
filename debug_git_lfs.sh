#!/bin/bash

# Start the interactive rebase for the last 14 commits
git rebase -i HEAD~6 --interactive

# The above command will open an editor for you to mark commits for editing.
# For each commit you want to amend, replace "pick" with "edit" and save the file.

# After saving, Git will start the rebase and stop at the first commit marked for editing.
# For each stopped commit, execute the following commands:

while true; do
  # Track the specific files with Git LFS
  git lfs track "pipeline_storage/docstore_2024-03-13.json"
  # git lfs track "src/local_chromium/chrome-linux64/chrome"

  # Check if files exist and add them if they do
  [ -f "pipeline_storage/docstore_2024-03-13.json" ] && git add "pipeline_storage/docstore_2024-03-13.json"
  # [ -f "src/local_chromium/chrome-linux64/chrome" ] && git add "src/local_chromium/chrome-linux64/chrome"

  # Always add .gitattributes, as it's updated by the git lfs track command
  git add .gitattributes

  # Amend the commit. If there's nothing to commit, this will fail, and the script will attempt to rebase --continue
  git commit --amend --no-edit || true

  # Continue the rebase. If there's no more commits to rebase, exit the loop
  git rebase --continue || break
done
