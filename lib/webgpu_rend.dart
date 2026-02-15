import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:webgpu_rend/src/webgpu_bindings_generated.dart';

export 'package:webgpu_rend/src/webgpu_bindings_generated.dart';

class WebgpuRend {
  static final WebgpuRend instance = WebgpuRend._();
  late final DynamicLibrary dylib;
  late final WebGpuBindings wgpu;

  // Glue Functions
  late final Pointer<Void> Function(int w, int h) _createTexture;
  late final int Function(Pointer<Void>) _getTextureId;
  late final Pointer<Void> Function(Pointer<Void>) _getWgpuView;
  late final Pointer<Void> Function(Pointer<Void>) _getWgpuTexture;
  late final void Function(Pointer<Void>) _beginAccess;
  late final void Function(Pointer<Void>) _endAccess;
  late final void Function(Pointer<Void>) _present;
  late final void Function(Pointer<Void>) _disposeTexture;

  // Raw pointer for NativeFinalizer
  late final Pointer<NativeFunction<Void Function(Pointer<Void>)>>
      _disposeTexturePtr;

  late final Pointer<Void> Function(Pointer<Void>) _init;
  late final Pointer<Void> Function(Pointer<Char>) _getProcAddress;

  late final WGPUDevice device;
  late final WGPUQueue queue;

  WebgpuRend._() {
    if (Platform.isWindows) {
      dylib = DynamicLibrary.open('webgpu_rend_plugin.dll');
    } else if (Platform.isAndroid) {
      dylib = DynamicLibrary.open('libwebgpu_rend_android.so');
    } else {
      throw "Unsupported Platform: ${Platform.operatingSystem}";
    }

    _createTexture = dylib
        .lookup<NativeFunction<Pointer<Void> Function(Int32, Int32)>>(
            'webgpu_rend_create_texture')
        .asFunction();
    _getTextureId = dylib
        .lookup<NativeFunction<Int64 Function(Pointer<Void>)>>(
            'webgpu_rend_get_texture_id')
        .asFunction();
    _getWgpuView = dylib
        .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void>)>>(
            'webgpu_rend_get_wgpu_texture_view')
        .asFunction();
    _getWgpuTexture = dylib
        .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void>)>>(
            'webgpu_rend_get_wgpu_texture')
        .asFunction();
    _beginAccess = dylib
        .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
            'webgpu_rend_texture_begin_access')
        .asFunction();
    _endAccess = dylib
        .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
            'webgpu_rend_texture_end_access')
        .asFunction();
    _present = dylib
        .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
            'webgpu_rend_present_texture')
        .asFunction();

    _disposeTexturePtr =
        dylib.lookup<NativeFunction<Void Function(Pointer<Void>)>>(
            'webgpu_rend_dispose_texture');
    _disposeTexture = _disposeTexturePtr.asFunction();

    _init = dylib
        .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void>)>>(
            'webgpu_rend_init')
        .asFunction();
    _getProcAddress = dylib
        .lookup<NativeFunction<Pointer<Void> Function(Pointer<Char>)>>(
            'webgpu_rend_get_proc_address')
        .asFunction();

    wgpu = WebGpuBindings.fromLookup(_lookupProc);
  }

  Pointer<T> _lookupProc<T extends NativeType>(String symbolName) {
    if (dylib.providesSymbol(symbolName)) return dylib.lookup(symbolName);
    final namePtr = symbolName.toNativeUtf8();
    final ptr = _getProcAddress(namePtr.cast());
    malloc.free(namePtr);
    if (ptr == nullptr) throw "Symbol not found: $symbolName";
    return ptr.cast();
  }

  Future<void> initialize() async {
    final rawDevicePtr = _init(nullptr);
    device = rawDevicePtr.cast();
    queue = wgpu.wgpuDeviceGetQueue(device);
  }

  Pointer<Void> createTextureInternal(int w, int h) => _createTexture(w, h);
  int getTextureIdInternal(Pointer<Void> handle) => _getTextureId(handle);
  Pointer<Void> getWgpuViewInternal(Pointer<Void> handle) =>
      _getWgpuView(handle);
  Pointer<Void> getWgpuTextureInternal(Pointer<Void> handle) =>
      _getWgpuTexture(handle);
  void beginAccessInternal(Pointer<Void> handle) => _beginAccess(handle);
  void endAccessInternal(Pointer<Void> handle) => _endAccess(handle);
  void presentInternal(Pointer<Void> handle) => _present(handle);
  void disposeTextureInternal(Pointer<Void> handle) => _disposeTexture(handle);
  Pointer<NativeFunction<Void Function(Pointer<Void>)>> get disposeTexturePtr =>
      _disposeTexturePtr;
}
