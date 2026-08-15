import 'dart:ffi';

import 'package:ffi/ffi.dart';

typedef _LastErrorNative = Pointer<Utf8> Function();
typedef _LastErrorDart = Pointer<Utf8> Function();
typedef _ByteOutputNative = Int32 Function(Pointer<Uint8>);
typedef _ByteOutputDart = int Function(Pointer<Uint8>);
typedef _Uint32OutputNative = Int32 Function(Pointer<Uint32>);
typedef _Uint32OutputDart = int Function(Pointer<Uint32>);
typedef _Uint64OutputNative = Int32 Function(Pointer<Uint64>);
typedef _Uint64OutputDart = int Function(Pointer<Uint64>);
typedef _ManualSeedNative = Int32 Function(Uint64);
typedef _ManualSeedDart = int Function(int);

typedef _TensorBoolTransformNative =
    Int32 Function(Uint64, Uint8, Pointer<Uint64>);
typedef _TensorBoolTransformDart = int Function(int, int, Pointer<Uint64>);
typedef _TensorBoolMetadataNative = Int32 Function(Uint64, Pointer<Uint8>);
typedef _TensorBoolMetadataDart = int Function(int, Pointer<Uint8>);
typedef _TensorStatusNative = Int32 Function(Uint64);
typedef _TensorStatusDart = int Function(int);
typedef _TensorUnaryNative = Int32 Function(Uint64, Pointer<Uint64>);
typedef _TensorUnaryDart = int Function(int, Pointer<Uint64>);
typedef _TensorBinaryNative = Int32 Function(Uint64, Uint64, Pointer<Uint64>);
typedef _TensorBinaryDart = int Function(int, int, Pointer<Uint64>);

typedef _LinearCreateNative =
    Int32 Function(Int64, Int64, Uint8, Pointer<Uint64>);
typedef _LinearCreateDart = int Function(int, int, int, Pointer<Uint64>);
typedef _ModuleForwardNative = Int32 Function(Uint64, Uint64, Pointer<Uint64>);
typedef _ModuleForwardDart = int Function(int, int, Pointer<Uint64>);
typedef _ModuleBoolNative = Int32 Function(Uint64, Uint8);
typedef _ModuleBoolDart = int Function(int, int);
typedef _ModuleToDeviceNative = Int32 Function(Uint64, Uint32, Int32);
typedef _ModuleToDeviceDart = int Function(int, int, int);
typedef _ModuleCountNative = Int32 Function(Uint64, Pointer<Size>);
typedef _ModuleCountDart = int Function(int, Pointer<Size>);
typedef _ModuleTensorAtNative = Int32 Function(Uint64, Size, Pointer<Uint64>);
typedef _ModuleTensorAtDart = int Function(int, int, Pointer<Uint64>);
typedef _ModulePathNative = Int32 Function(Uint64, Pointer<Utf8>);
typedef _ModulePathDart = int Function(int, Pointer<Utf8>);
typedef _HandleReleaseNative = Int32 Function(Uint64);
typedef _HandleReleaseDart = int Function(int);

typedef _SgdCreateNative =
    Int32 Function(Uint64, Double, Double, Double, Pointer<Uint64>);
typedef _SgdCreateDart =
    int Function(int, double, double, double, Pointer<Uint64>);
typedef _AdamCreateNative =
    Int32 Function(
      Uint64,
      Double,
      Double,
      Double,
      Double,
      Double,
      Pointer<Uint64>,
    );
typedef _AdamCreateDart =
    int Function(int, double, double, double, double, double, Pointer<Uint64>);

typedef _ParameterSgdCreateNative =
    Int32 Function(
      Pointer<Uint64>,
      Size,
      Double,
      Double,
      Double,
      Pointer<Uint64>,
    );
typedef _ParameterSgdCreateDart =
    int Function(
      Pointer<Uint64>,
      int,
      double,
      double,
      double,
      Pointer<Uint64>,
    );
typedef _ParameterAdamCreateNative =
    Int32 Function(
      Pointer<Uint64>,
      Size,
      Double,
      Double,
      Double,
      Double,
      Double,
      Pointer<Uint64>,
    );
typedef _ParameterAdamCreateDart =
    int Function(
      Pointer<Uint64>,
      int,
      double,
      double,
      double,
      double,
      double,
      Pointer<Uint64>,
    );

final class NativeTrainingBindings {
  NativeTrainingBindings(DynamicLibrary library)
    : lastErrorMessage = library
          .lookupFunction<_LastErrorNative, _LastErrorDart>(
            'ts_last_error_message',
          ),
      trainingAvailable = library
          .lookupFunction<_ByteOutputNative, _ByteOutputDart>(
            'ts_training_available',
          ),
      cudaDeviceCount = library
          .lookupFunction<_Uint32OutputNative, _Uint32OutputDart>(
            'ts_runtime_cuda_device_count',
          ),
      manualSeed = library.lookupFunction<_ManualSeedNative, _ManualSeedDart>(
        'ts_manual_seed',
      ),
      tensorWithRequiresGrad = library
          .lookupFunction<_TensorBoolTransformNative, _TensorBoolTransformDart>(
            'ts_tensor_with_requires_grad',
          ),
      tensorRequiresGrad = library
          .lookupFunction<_TensorBoolMetadataNative, _TensorBoolMetadataDart>(
            'ts_tensor_requires_grad',
          ),
      tensorBackward = library
          .lookupFunction<_TensorStatusNative, _TensorStatusDart>(
            'ts_tensor_backward',
          ),
      tensorGrad = library.lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
        'ts_tensor_grad',
      ),
      tensorRelu = library.lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
        'ts_tensor_relu',
      ),
      tensorSigmoid = library
          .lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
            'ts_tensor_sigmoid',
          ),
      tensorTanh = library.lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
        'ts_tensor_tanh',
      ),
      tensorGelu = library.lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
        'ts_tensor_gelu',
      ),
      tensorSilu = library.lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
        'ts_tensor_silu',
      ),
      tensorSwiGlu = library
          .lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
            'ts_tensor_swiglu',
          ),
      mseLoss = library.lookupFunction<_TensorBinaryNative, _TensorBinaryDart>(
        'ts_mse_loss',
      ),
      crossEntropyLoss = library
          .lookupFunction<_TensorBinaryNative, _TensorBinaryDart>(
            'ts_cross_entropy_loss',
          ),
      linearCreate = library
          .lookupFunction<_LinearCreateNative, _LinearCreateDart>(
            'ts_linear_create',
          ),
      moduleForward = library
          .lookupFunction<_ModuleForwardNative, _ModuleForwardDart>(
            'ts_module_forward',
          ),
      moduleSetTraining = library
          .lookupFunction<_ModuleBoolNative, _ModuleBoolDart>(
            'ts_module_set_training',
          ),
      moduleToDevice = library
          .lookupFunction<_ModuleToDeviceNative, _ModuleToDeviceDart>(
            'ts_module_to_device',
          ),
      moduleParameterCount = library
          .lookupFunction<_ModuleCountNative, _ModuleCountDart>(
            'ts_module_parameter_count',
          ),
      moduleParameterAt = library
          .lookupFunction<_ModuleTensorAtNative, _ModuleTensorAtDart>(
            'ts_module_parameter_at',
          ),
      moduleBufferCount = library
          .lookupFunction<_ModuleCountNative, _ModuleCountDart>(
            'ts_module_buffer_count',
          ),
      moduleBufferAt = library
          .lookupFunction<_ModuleTensorAtNative, _ModuleTensorAtDart>(
            'ts_module_buffer_at',
          ),
      moduleSave = library.lookupFunction<_ModulePathNative, _ModulePathDart>(
        'ts_module_save',
      ),
      moduleLoad = library.lookupFunction<_ModulePathNative, _ModulePathDart>(
        'ts_module_load',
      ),
      moduleRelease = library
          .lookupFunction<_HandleReleaseNative, _HandleReleaseDart>(
            'ts_module_release',
          ),
      sgdCreate = library.lookupFunction<_SgdCreateNative, _SgdCreateDart>(
        'ts_sgd_create',
      ),
      adamCreate = library.lookupFunction<_AdamCreateNative, _AdamCreateDart>(
        'ts_adam_create',
      ),
      adamWCreate = library.lookupFunction<_AdamCreateNative, _AdamCreateDart>(
        'ts_adamw_create',
      ),
      optimizerZeroGrad = library
          .lookupFunction<_HandleReleaseNative, _HandleReleaseDart>(
            'ts_optimizer_zero_grad',
          ),
      optimizerStep = library
          .lookupFunction<_HandleReleaseNative, _HandleReleaseDart>(
            'ts_optimizer_step',
          ),
      optimizerRelease = library
          .lookupFunction<_HandleReleaseNative, _HandleReleaseDart>(
            'ts_optimizer_release',
          ),
      parameterSgdCreate = library
          .lookupFunction<_ParameterSgdCreateNative, _ParameterSgdCreateDart>(
            'ts_sgd_create_for_tensors',
          ),
      parameterAdamCreate = library
          .lookupFunction<_ParameterAdamCreateNative, _ParameterAdamCreateDart>(
            'ts_adam_create_for_tensors',
          ),
      parameterAdamWCreate = library
          .lookupFunction<_ParameterAdamCreateNative, _ParameterAdamCreateDart>(
            'ts_adamw_create_for_tensors',
          ),
      parameterOptimizerZeroGrad = library
          .lookupFunction<_HandleReleaseNative, _HandleReleaseDart>(
            'ts_parameter_optimizer_zero_grad',
          ),
      parameterOptimizerStep = library
          .lookupFunction<_HandleReleaseNative, _HandleReleaseDart>(
            'ts_parameter_optimizer_step',
          ),
      parameterOptimizerRelease = library
          .lookupFunction<_HandleReleaseNative, _HandleReleaseDart>(
            'ts_parameter_optimizer_release',
          ),
      liveModuleCount = library
          .lookupFunction<_Uint64OutputNative, _Uint64OutputDart>(
            'ts_runtime_live_module_count',
          ),
      liveOptimizerCount = library
          .lookupFunction<_Uint64OutputNative, _Uint64OutputDart>(
            'ts_runtime_live_optimizer_count',
          );

  final _LastErrorDart lastErrorMessage;
  final _ByteOutputDart trainingAvailable;
  final _Uint32OutputDart cudaDeviceCount;
  final _ManualSeedDart manualSeed;
  final _TensorBoolTransformDart tensorWithRequiresGrad;
  final _TensorBoolMetadataDart tensorRequiresGrad;
  final _TensorStatusDart tensorBackward;
  final _TensorUnaryDart tensorGrad;
  final _TensorUnaryDart tensorRelu;
  final _TensorUnaryDart tensorSigmoid;
  final _TensorUnaryDart tensorTanh;
  final _TensorUnaryDart tensorGelu;
  final _TensorUnaryDart tensorSilu;
  final _TensorUnaryDart tensorSwiGlu;
  final _TensorBinaryDart mseLoss;
  final _TensorBinaryDart crossEntropyLoss;
  final _LinearCreateDart linearCreate;
  final _ModuleForwardDart moduleForward;
  final _ModuleBoolDart moduleSetTraining;
  final _ModuleToDeviceDart moduleToDevice;
  final _ModuleCountDart moduleParameterCount;
  final _ModuleTensorAtDart moduleParameterAt;
  final _ModuleCountDart moduleBufferCount;
  final _ModuleTensorAtDart moduleBufferAt;
  final _ModulePathDart moduleSave;
  final _ModulePathDart moduleLoad;
  final _HandleReleaseDart moduleRelease;
  final _SgdCreateDart sgdCreate;
  final _AdamCreateDart adamCreate;
  final _AdamCreateDart adamWCreate;
  final _HandleReleaseDart optimizerZeroGrad;
  final _HandleReleaseDart optimizerStep;
  final _HandleReleaseDart optimizerRelease;
  final _ParameterSgdCreateDart parameterSgdCreate;
  final _ParameterAdamCreateDart parameterAdamCreate;
  final _ParameterAdamCreateDart parameterAdamWCreate;
  final _HandleReleaseDart parameterOptimizerZeroGrad;
  final _HandleReleaseDart parameterOptimizerStep;
  final _HandleReleaseDart parameterOptimizerRelease;
  final _Uint64OutputDart liveModuleCount;
  final _Uint64OutputDart liveOptimizerCount;
}
