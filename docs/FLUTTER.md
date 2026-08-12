# Tensora for Flutter

Flutter is a first-class Tensora deployment environment, not an adapter added after the core framework is complete.

This document defines the intended integration model.

## 1. Goal

A Flutter developer should be able to load and execute a supported local model without writing application-level Kotlin, Swift, Objective-C, C++, JNI, or custom FFI glue.

Target experience:

```dart
final model = await TensoraModel.asset(
  'assets/model.tmodel',
);

final result = await model(input);
```

Tensora should own the native integration necessary for supported workflows.

## 2. Package boundary

Flutter-specific behavior belongs in:

```text
tensora_flutter
```

The dependency direction is:

```text
tensora_flutter
      ↓
Tensora core packages
      ↓
native runtime
```

Core Tensora packages must never depend on Flutter.

This keeps server, CLI, desktop Dart, and test environments independent from Flutter.

## 3. Flutter responsibilities

`tensora_flutter` may provide:

- model asset loading;
- lifecycle-aware controllers;
- camera adapters;
- audio adapters;
- optional widgets;
- platform plugin integration;
- platform/device diagnostics;
- ergonomic conversion between Flutter-facing media types and Tensora runtime inputs.

It must not define core tensor semantics or numerical kernels.

## 4. UI isolate rule

Heavy inference, model loading, preprocessing, or compilation must not block Flutter's UI isolate by default.

Conceptual path:

```text
Flutter UI
   ↓
controller / async request
   ↓
Tensora runtime bridge
   ↓
native worker/provider
   ↓
CPU/GPU/NPU
```

The UI isolate should receive small typed results/events rather than perform the numerical workload itself.

## 5. Lifecycle

Every Flutter runtime component must define behavior for:

- widget disposal;
- application pause/resume;
- background/foreground;
- platform view/camera teardown;
- audio interruption;
- permission loss;
- device/backend failure;
- memory pressure where platform APIs expose it;
- cancellation while work is running.

Lifecycle cleanup must be deterministic for large native/device resources.

## 6. Camera architecture

Realtime vision is a flagship Tensora workflow.

Preferred architecture:

```text
Camera/native frame
        ↓
platform frame adapter
        ↓
native resize/color conversion
        ↓
Tensora Tensor
        ↓
model inference
        ↓
postprocessing
        ↓
small typed Dart result
        ↓
Flutter overlay
```

Avoid this path when a safer direct native path is available:

```text
camera frame
 ↓
copy full frame to Dart bytes
 ↓
copy full frame back to native
 ↓
model
```

Copies that cannot be avoided should be measurable in profiling.

## 7. Camera backpressure

A camera may produce frames faster than a model can consume them.

Realtime pipelines require bounded buffering.

Planned policies:

```text
latest
skip/drop oldest
block where appropriate
```

The default for interactive live vision should normally prioritize the newest frame.

Never permit an unbounded frame queue.

## 8. Camera controller

Possible API direction:

```dart
final controller = VisionController(
  model: model,
  targetFps: 30,
  queuePolicy: QueuePolicy.latest,
);
```

The controller owns runtime state, scheduling, cancellation, and lifecycle coordination.

Widgets should be thin views over controllers.

## 9. Optional widgets

Potential convenience widgets:

```text
TensoraCameraView
DetectionOverlay
SegmentationOverlay
PoseOverlay
InferenceBuilder
LocalLLMBuilder
EmbeddingBuilder
```

High-level widgets are optional. Core functionality must remain usable without them.

## 10. Typed results

Return small domain-level results when practical.

Example:

```dart
Detection(
  label: 'car',
  confidence: 0.97,
  boundingBox: ...,
)
```

Do not force UI code to understand raw provider tensors for common task APIs.

Raw tensor APIs remain available for advanced users through core Tensora.

## 11. Audio

Streaming audio should use fixed-size/ring buffers and avoid repeated large allocation.

Potential uses:

- voice activity detection;
- speech recognition;
- keyword spotting;
- speaker embeddings;
- audio classification.

The audio pipeline must define ownership when the platform interrupts or replaces an audio session.

## 12. Local language models

A Flutter local language-model API should separate model weights from session state.

Conceptual API:

```dart
final model = await LocalLanguageModel.load(
  'assets/model.tmodel',
);

final session = model.createSession();

await for (final token in session.generateStream(prompt)) {
  // update UI
}
```

The session owns mutable generation state such as KV cache.

Generation must support cancellation and deterministic cleanup.

## 13. Asset loading

The Flutter package should support packaged assets without requiring users to understand native filesystem layout.

Large model handling must consider:

- copy-on-first-use behavior;
- mmap/direct loading where possible;
- app bundle constraints;
- writable cache locations;
- model versioning;
- integrity validation.

The implementation should avoid unnecessary duplicate storage for large models where platform rules allow.

## 14. Model updates

Future optional model distribution may support model updates independently of application binaries where platform policies allow.

Update flow must be safe:

```text
download candidate
 ↓
verify
 ↓
compatibility check
 ↓
golden validation
 ↓
atomic activation
 ↓
retain rollback version
```

Core Flutter inference must not require remote model distribution.

## 15. Execution policy

Flutter workloads need policy-based execution because battery, memory, thermals, and responsiveness matter.

Potential API:

```dart
ExecutionPolicy(
  mode: PerformanceMode.balanced,
  maxMemoryMb: 1024,
  targetLatencyMs: 50,
)
```

Policies may influence provider/device choice, batching, or supported adaptive behaviors.

Any adaptation must be observable and must not silently change semantic correctness.

## 16. Thermal behavior

Where reliable platform APIs exist, optional policies may react to thermal pressure by adjusting non-semantic performance parameters such as:

- target inference FPS;
- frame sampling;
- model variant selected by the application;
- batch size;
- execution provider.

Tensora must not pretend it can guarantee a universal temperature or FPS target across devices.

## 17. Memory pressure

The runtime should be able to release disposable caches where safe when the platform reports pressure.

Do not release resources whose loss would violate active operation semantics without coordinating cancellation/recovery.

## 18. Platform permissions

Tensora should not silently request unrelated permissions.

Camera/audio integrations should surface permission errors through normal Flutter/plugin semantics and document required platform configuration.

## 19. Privacy

Local inference remains local by default.

`tensora_flutter` must not upload camera frames, audio, prompts, embeddings, or inference results unless the application explicitly uses a separate networked feature.

## 20. Binary size

Mobile applications should only package native backends they need where packaging technology allows.

Do not force full training runtimes into an application that only needs lightweight inference.

Track binary-size impact as part of backend integration decisions.

## 21. Startup and warmup

Measure separately:

- native runtime initialization;
- model asset preparation;
- model load;
- provider initialization;
- first inference;
- warm inference.

Support explicit warmup where useful:

```dart
await model.warmup();
```

## 22. Diagnostics

Flutter diagnostics should eventually expose:

```text
selected backend/device
model load time
inference latency
p50/p95/p99 latency
camera input FPS
inference FPS
dropped frames
queue depth
memory usage
host/device transfers
thermal state where available
```

## 23. Testing gate

A Flutter feature is not complete until it is tested for relevant:

- Dart behavior;
- plugin/native integration;
- lifecycle transitions;
- repeated initialization/disposal;
- cancellation;
- memory stability;
- representative real devices;
- release/profile-mode performance.

## 24. First Flutter vertical slice

The first production-quality Flutter proof should be:

```text
Flutter application
 ↓
asset `.tmodel`
 ↓
Tensora runtime
 ↓
portable inference backend
 ↓
typed prediction
 ↓
UI
```

The second flagship slice should add live camera inference with bounded buffering and stable lifecycle behavior.

## 25. Success condition

Tensora succeeds on Flutter when a developer can treat local AI as a normal application capability rather than a separate native integration project, while still retaining control over performance, memory, device selection, privacy, and failure behavior.