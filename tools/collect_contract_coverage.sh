#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$root_dir/build/libtensora_contract_fixture.so"

mkdir -p "$root_dir/build"

cc -std=c11 -fPIC -shared -Wall -Wextra -Werror \
  -DTENSORA_NATIVE_BUILD \
  -Dts_test_set_training_mode=ts_fixture_base_set_training_mode \
  -Dts_module_parameter_count=ts_fixture_base_module_parameter_count \
  -I "$root_dir/native/include" \
  "$root_dir/native/tests/dart_contract_fixture.c" \
  "$root_dir/native/tests/dart_contract_nn_v2_fixture.c" \
  -o "$fixture"

cd "$root_dir/packages/tensora"
TENSORA_NATIVE_LIBRARY="$fixture" \
  dart test integration_test/native_contract_fixture_test.dart \
  --coverage-path=coverage/native-contract.info \
  --reporter expanded

TENSORA_NATIVE_LIBRARY="$fixture" \
  dart test integration_test/native_contract_completion_test.dart \
  --coverage-path=coverage/native-contract-completion.info \
  --reporter expanded
