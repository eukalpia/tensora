# Tensora API Design Guidelines

Tensora's public API is a long-lived compatibility surface. The implementation behind it may change substantially as native providers, compilers, and device backends evolve. The API must therefore model Tensora concepts rather than current implementation details.

## 1. Design goals

Public APIs should be:

- idiomatic Dart;
- strongly typed;
- discoverable;
- composable;
- backend-neutral;
- explicit about async behavior;
- explicit about disposal/resource ownership;
- useful outside Flutter unless intentionally Flutter-specific;
- difficult to misuse silently.

## 2. Provider neutrality

Prefer:

```dart
final tensor = Tensor.randn(
  [32, 512],
  device: Device.cuda(0),
);
```

Avoid public APIs shaped around implementation providers.

Provider-specific details belong in diagnostics or expert-only extensions, not the main stable programming model.

## 3. Naming

Use full, conventional ML terms unless a shorter form is overwhelmingly standard.

Prefer:

```text
Tensor
Device
Shape
Parameter
Optimizer
DataLoader
ExecutionPolicy
```

Avoid project-specific abbreviations that require memorization.

## 4. Constructors and factories

Use constructors for cheap, deterministic object construction when no asynchronous/native initialization is required.

Use async factories for operations such as:

- model loading;
- provider initialization;
- device probing requiring native work;
- compilation;
- large asset preparation.

Example:

```dart
final model = await Model.load('model.tmodel');
```

Do not hide expensive blocking initialization in innocent property access.

## 5. Async semantics

Potentially long-running user-facing operations should return `Future` or `Stream` as appropriate.

Examples:

```dart
final result = await model.run(input);

await for (final token in session.generateStream(prompt)) {
  ...
}
```

Document when completion means device work is actually complete versus merely scheduled.

## 6. Cancellation

Long-running operations should support cancellation where the backend/runtime can honor it safely.

Cancellation APIs must define:

- whether cancellation is best-effort;
- when resources are released;
- whether partial output can be emitted;
- session usability after cancellation.

## 7. Resource ownership

Types owning substantial native resources should provide explicit cleanup.

Possible pattern:

```dart
await model.dispose();
```

or synchronous disposal when safe.

Finalizers are a fallback, not the primary lifecycle API for expensive resources.

## 8. Immutable metadata

Prefer immutable value objects for:

- `Shape`;
- `DType` descriptors;
- device capability snapshots;
- model metadata;
- typed prediction results.

Immutability improves reasoning across asynchronous boundaries.

`DType` is the single public source of truth for element byte width, stable C
ABI value, numerical category, promotion, and reduction accumulator. A dtype
descriptor may exist before a native storage/backend implementation does;
factories and operators must reject unsupported combinations explicitly rather
than silently converting to `float32`.

## 9. Tensor mutation

Default tensor operations should favor clear functional semantics.

If in-place operations are offered, make them visibly distinct and document interaction with:

- autograd;
- views/aliasing;
- shared references;
- concurrency.

Do not introduce in-place mutation merely to imitate another framework.

## 10. Error types

Use typed Tensora exceptions for recoverable failures.

Example hierarchy direction:

```text
TensoraException
├── InvalidArgumentException
├── InvalidShapeException
├── BackendUnavailableException
├── UnsupportedOperationException
├── OutOfMemoryException
├── DeviceLostException
└── ModelFormatException
```

Errors should include useful structured context without requiring callers to parse provider strings.

## 11. Capability checks

Do not rely on exceptions as the only way to discover device support.

Provide capability queries where practical:

```dart
final caps = await device.capabilities();
```

But execution must still validate at runtime because installed provider capabilities can change or be model-specific.

## 12. Defaults

Defaults should be safe and unsurprising.

Examples:

- no silent lossy dtype conversion;
- no unbounded queue;
- no implicit remote networking;
- no hidden execution-provider fallback when strict semantics are expected;
- no arbitrary native extension loading.

Performance-oriented defaults must remain observable.

## 13. `bestAvailable`

Convenience device/provider selection is useful, but must remain inspectable.

Applications should be able to retrieve why a backend/device was selected and choose strict configuration when deterministic deployment matters.

## 14. Flutter separation

Flutter widget types never appear in core Tensor/NN APIs.

`tensora_flutter` may adapt core concepts into controllers and widgets.

This separation keeps core packages usable in:

- Dart CLI;
- server applications;
- tests;
- non-Flutter desktop contexts.

## 15. Task-specific high-level APIs

Tensora may provide high-level task APIs such as:

```text
ObjectDetector
EmbeddingModel
LocalLanguageModel
```

These should return typed application-level results while retaining a lower-level Tensor/model API for advanced use.

Task APIs should be adapters, not parallel incompatible runtimes.

## 16. Sessions

Use sessions when mutable per-request/per-conversation state must be separated from immutable/shared model state.

Example:

```dart
final session = model.createSession();
```

This pattern is particularly important for:

- language-model KV cache;
- streaming audio state;
- recurrent state;
- request-specific buffers.

## 17. Options objects

Use named parameters for small stable option sets.

Use immutable configuration objects when options become complex or reused across calls.

Avoid methods with long lists of positional primitives.

## 18. Experimental APIs

Unstable APIs should live in clearly marked experimental namespaces/packages and should not be accidentally documented as stable.

Graduating an API to stable requires:

- real usage;
- tests;
- coherent error/lifecycle semantics;
- compatibility review.

## 19. Deprecation

Stable APIs should be deprecated before removal when feasible.

Deprecation messages should identify the replacement and migration path.

## 20. Generated APIs

Repetitive operator APIs may be generated from a machine-readable operator registry.

Generated public APIs must still meet normal naming/documentation/compatibility standards.

The generator is not an excuse for exposing provider-specific complexity.

## 21. Public API review checklist

Before merging a new stable public symbol, ask:

1. Is the name idiomatic Dart?
2. Does it model a Tensora concept rather than one provider?
3. Can it work outside Flutter if appropriate?
4. Is expensive work visibly async?
5. Is disposal obvious?
6. Are failure modes typed?
7. Are defaults safe?
8. Can a new backend implement the same semantics?
9. Does this force a future compiler/runtime design unnecessarily?
10. Can this be supported for a full stable major-version lifecycle?

If several answers are uncertain, keep the API internal or experimental until the design is mature.