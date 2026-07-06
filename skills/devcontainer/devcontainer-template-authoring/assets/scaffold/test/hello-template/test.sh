#!/bin/sh
# Runs INSIDE the applied template's container (via the smoke-test
# action). Assert that what the template promises actually works.
set -e

echo "user: $(id -un)"
git --version
echo "hello-template smoke test passed"
