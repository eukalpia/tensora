/// Training orchestration for Tensora.
///
/// [Trainer] owns the epoch/batch loop and the lifetime of every Tensor a step
/// allocates. [LearningRateSchedule] and its implementations are pure step
/// arithmetic; nothing applies them to an optimizer yet, because the native ABI
/// has no way to change the learning rate of a live optimizer. See the
/// [LearningRateSchedule] documentation.
library;

export 'package:tensora/tensora.dart' show Device, Tensor, TensoraRuntime;
export 'package:tensora_nn/tensora_nn.dart';
export 'package:tensora_optim/tensora_optim.dart';

export 'src/history.dart';
export 'src/schedule.dart';
export 'src/trainer.dart';
