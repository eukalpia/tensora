#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$root_dir/build/libtensora_contract_fixture.so"

mkdir -p "$root_dir/build"

cc -std=c11 -fPIC -shared -Wall -Wextra -Werror \
  -DTENSORA_NATIVE_BUILD \
  -I "$root_dir/native/include" \
  "$root_dir/native/tests/dart_contract_fixture.c" \
  -o "$fixture"

cd "$root_dir/packages/tensora"
TENSORA_NATIVE_LIBRARY="$fixture" \
  dart test integration_test/native_contract_fixture_test.dart \
  --coverage-path=coverage/native-contract.info \
  --reporter expanded
