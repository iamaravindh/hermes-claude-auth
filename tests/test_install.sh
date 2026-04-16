#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAKE_HOME="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() {
    rm -rf "$FAKE_HOME"
}
trap cleanup EXIT

pass() {
    printf '[PASS] %s\n' "$1"
    PASS=$((PASS + 1))
}

fail() {
    printf '[FAIL] %s - %s\n' "$1" "$2"
    FAIL=$((FAIL + 1))
}

assert_file_exists() {
    local label="$1" path="$2"
    if [ -f "$path" ]; then
        return 0
    else
        fail "$label" "expected file not found: $path"
        return 1
    fi
}

assert_file_contains() {
    local label="$1" path="$2" needle="$3"
    if grep -qF "$needle" "$path" 2>/dev/null; then
        return 0
    else
        fail "$label" "file $path does not contain: $needle"
        return 1
    fi
}

assert_file_not_exists() {
    local label="$1" path="$2"
    if [ ! -e "$path" ]; then
        return 0
    else
        fail "$label" "expected absent but found: $path"
        return 1
    fi
}

assert_dir_not_exists() {
    local label="$1" path="$2"
    if [ ! -d "$path" ]; then
        return 0
    else
        fail "$label" "expected absent dir but found: $path"
        return 1
    fi
}

export HOME="$FAKE_HOME"

mkdir -p "$FAKE_HOME/.hermes/hermes-agent"
python3 -m venv "$FAKE_HOME/.hermes/hermes-agent/venv"

VENV_PYTHON="$FAKE_HOME/.hermes/hermes-agent/venv/bin/python"
SITE_PACKAGES="$("$VENV_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
SITECUSTOMIZE="$SITE_PACKAGES/sitecustomize.py"
BACKUP="$SITECUSTOMIZE.pre-hermes-claude-auth"
PATCH_FILE="$FAKE_HOME/.hermes/patches/good.py"

# Test 1: Fresh install
T1="Test 1: Fresh install"
if "$REPO_DIR/install.sh" >/dev/null 2>&1; then
    ok=1
    assert_file_exists "$T1" "$PATCH_FILE" || ok=0
    assert_file_exists "$T1" "$SITECUSTOMIZE" || ok=0
    assert_file_contains "$T1" "$SITECUSTOMIZE" "# hermes-claude-auth managed" || ok=0
    [ "$ok" -eq 1 ] && pass "$T1"
else
    fail "$T1" "install.sh exited non-zero"
fi

# Test 2: Idempotent re-install
T2="Test 2: Idempotent re-install"
if "$REPO_DIR/install.sh" >/dev/null 2>&1; then
    ok=1
    assert_file_exists "$T2" "$SITECUSTOMIZE" || ok=0
    assert_file_contains "$T2" "$SITECUSTOMIZE" "# hermes-claude-auth managed" || ok=0
    count="$(grep -cF '# hermes-claude-auth managed' "$SITECUSTOMIZE" 2>/dev/null || true)"
    if [ "$count" -gt 1 ]; then
        fail "$T2" "marker duplicated ($count occurrences)"
        ok=0
    fi
    [ "$ok" -eq 1 ] && pass "$T2"
else
    fail "$T2" "install.sh exited non-zero on re-run"
fi

# Test 3: Install over existing sitecustomize.py (no marker)
T3="Test 3: Install over existing sitecustomize.py"
printf 'import sys\n# some unrelated hook\n' > "$SITECUSTOMIZE"
if "$REPO_DIR/install.sh" >/dev/null 2>&1; then
    ok=1
    assert_file_exists "$T3" "$BACKUP" || ok=0
    assert_file_contains "$T3" "$SITECUSTOMIZE" "# hermes-claude-auth managed" || ok=0
    assert_file_contains "$T3" "$BACKUP" "# some unrelated hook" || ok=0
    [ "$ok" -eq 1 ] && pass "$T3"
else
    fail "$T3" "install.sh exited non-zero"
fi

# Test 4: Uninstall (hook only)
T4="Test 4: Uninstall (hook only)"
if "$REPO_DIR/uninstall.sh" >/dev/null 2>&1; then
    ok=1
    assert_file_exists "$T4" "$SITECUSTOMIZE" || ok=0
    assert_file_contains "$T4" "$SITECUSTOMIZE" "# some unrelated hook" || ok=0
    assert_file_not_exists "$T4" "$BACKUP" || ok=0
    assert_file_exists "$T4" "$PATCH_FILE" || ok=0
    [ "$ok" -eq 1 ] && pass "$T4"
else
    fail "$T4" "uninstall.sh exited non-zero"
fi

# Test 5: Reinstall then uninstall --purge
T5="Test 5: Reinstall then uninstall --purge"
rm -f "$SITECUSTOMIZE"
if "$REPO_DIR/install.sh" >/dev/null 2>&1 && "$REPO_DIR/uninstall.sh" --purge >/dev/null 2>&1; then
    ok=1
    assert_file_not_exists "$T5" "$SITECUSTOMIZE" || ok=0
    assert_file_not_exists "$T5" "$PATCH_FILE" || ok=0
    assert_dir_not_exists "$T5" "$FAKE_HOME/.hermes/patches" || ok=0
    [ "$ok" -eq 1 ] && pass "$T5"
else
    fail "$T5" "install.sh or uninstall.sh --purge exited non-zero"
fi

TOTAL=$((PASS + FAIL))
printf '\n%d/%d tests passed\n' "$PASS" "$TOTAL"
[ "$FAIL" -eq 0 ]
