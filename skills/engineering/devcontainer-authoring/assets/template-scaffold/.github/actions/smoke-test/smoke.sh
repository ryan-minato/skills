#!/usr/bin/env bash
# Smoke-test one template: substitute default option values into a copy of
# src/<id> (what `devcontainer templates apply` would do), start the dev
# container, and run the template's test script inside it.
set -euo pipefail

TEMPLATE_ID="${1:?usage: smoke.sh <template-id>}"

SRC_DIR=$(mktemp -d "/tmp/${TEMPLATE_ID}.XXXXXX")
ID_LABEL="smoke-test=${TEMPLATE_ID}-$$"

cleanup() {
    containers=$(docker container ls -q -f "label=${ID_LABEL}")
    [ -n "$containers" ] && docker rm -f $containers >/dev/null 2>&1 || true
    rm -rf "${SRC_DIR}"
}
trap cleanup EXIT

shopt -s dotglob
cp -R "src/${TEMPLATE_ID}/." "${SRC_DIR}/"

# Replace every ${templateOption:<key>} with the option's default value.
# A missing default is an authoring bug: apply must work with no input.
manifest="${SRC_DIR}/devcontainer-template.json"
for option in $(jq -r '.options // {} | keys[]' "$manifest"); do
    value=$(jq -r --arg o "$option" '.options[$o].default // empty' "$manifest")
    if [ -z "$value" ]; then
        echo "template '${TEMPLATE_ID}': option '${option}' has no default" >&2
        exit 1
    fi
    # Escape for the sed REPLACEMENT context: backslash first, then the
    # special replacement characters & and the / delimiter.
    escaped=${value//\\/\\\\}
    escaped=${escaped//&/\\&}
    escaped=${escaped//\//\\/}
    find "${SRC_DIR}" -type f -print0 \
        | xargs -0 sed -i "s/\${templateOption:${option}}/${escaped}/g"
done

# Ship the template's test script into the workspace the container mounts.
if [ -d "test/${TEMPLATE_ID}" ]; then
    mkdir -p "${SRC_DIR}/test-project"
    cp -Rp "test/${TEMPLATE_ID}/." "${SRC_DIR}/test-project/"
fi

echo "(*) Starting dev container for '${TEMPLATE_ID}'"
devcontainer up --id-label "${ID_LABEL}" --workspace-folder "${SRC_DIR}"

echo "(*) Running test script"
devcontainer exec --workspace-folder "${SRC_DIR}" --id-label "${ID_LABEL}" \
    /bin/sh -c 'set -e
        if [ -f test-project/test.sh ]; then
            cd test-project && sh ./test.sh
        else
            echo "no test-project/test.sh - container started OK"
        fi'
