# WebgpuRend

Flutter bindings for WebGPU via https://github.com/google/dawn. Currently supports Windows and Android.

# Demo

Download the android apk from the [releases](https://github.com/jacksonrl/flutter_webgpu_rend/releases) tab, which uses the examples found in the [example](https://github.com/jacksonrl/flutter_webgpu_rend/tree/master/example) project.

<img width="637" height="585" alt="image" src="https://github.com/user-attachments/assets/2a4a0103-a42d-4414-b145-a1ecf8c5b74f" />


# Usage

Simple usage:

```dart

import 'dart:async';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:webgpu_rend/webgpu_rend.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await WebgpuRend.instance.initialize();
  
  runApp(const MaterialApp(
    home: RawWebGpuTriangle(),
    debugShowCheckedModeBanner: false,
  ));
}

const String kShaderWgsl = r'''
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5)
    );
    return vec4<f32>(pos[in_vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.2, 0.3, 1.0); // Reddish triangle
}
''';

class RawWebGpuTriangle extends StatefulWidget {
  const RawWebGpuTriangle({super.key});

  @override
  State<RawWebGpuTriangle> createState() => _RawWebGpuTriangleState();
}

class _RawWebGpuTriangleState extends State<RawWebGpuTriangle> with SingleTickerProviderStateMixin {
  // Texture/Interop handles
  Pointer<Void>? _textureHandle; 
  int _textureId = -1;

  // WebGPU specific typed pointers
  WGPUTextureView? _wgpuTextureView; 
  WGPURenderPipeline? _renderPipeline;
  
  late Ticker _ticker;
  bool _isReady = false;
  
  final int _width = 500;
  final int _height = 500;

  WebGpuBindings get _wgpu => WebgpuRend.instance.wgpu;
  WGPUDevice get _device => WebgpuRend.instance.device;
  WGPUQueue get _queue => WebgpuRend.instance.queue;

  @override
  void initState() {
    super.initState();
    _initRawWebGpu();
  }

  Future<void> _initRawWebGpu() async {
    // Create Texture via plugin C++ interop
    _textureHandle = WebgpuRend.instance.createTextureInternal(_width, _height);
    _textureId = WebgpuRend.instance.getTextureIdInternal(_textureHandle!);
    
    // Extract the WGPU View and cast it to the correct FFI type
    final rawViewPtr = WebgpuRend.instance.getWgpuViewInternal(_textureHandle!);
    _wgpuTextureView = rawViewPtr.cast<WGPUTextureViewImpl>(); 

    using((arena) {
      // Shader Module Setup
      final nativeSource = kShaderWgsl.toNativeUtf8(allocator: arena);
      final shaderCodeDesc = arena<WGPUShaderSourceWGSL>();
      shaderCodeDesc.ref.chain.sType = WGPUSType.WGPUSType_ShaderSourceWGSL;
      shaderCodeDesc.ref.code.data = nativeSource.cast();
      shaderCodeDesc.ref.code.length = kShaderWgsl.length;

      final shaderModuleDesc = arena<WGPUShaderModuleDescriptor>();
      shaderModuleDesc.ref.nextInChain = shaderCodeDesc.cast();
      final shaderModule = _wgpu.wgpuDeviceCreateShaderModule(_device, shaderModuleDesc);

      // Pipeline Setup
      final textureFormat = Platform.isAndroid
          ? WGPUTextureFormat.WGPUTextureFormat_RGBA8Unorm
          : WGPUTextureFormat.WGPUTextureFormat_BGRA8Unorm;

      final colorTarget = arena<WGPUColorTargetState>();
      colorTarget.ref.format = textureFormat;
      colorTarget.ref.writeMask = WGPUColorWriteMask_All;

      final fragmentState = arena<WGPUFragmentState>();
      fragmentState.ref.module = shaderModule;
      fragmentState.ref.entryPoint = _createStringView(arena, "fs_main");
      fragmentState.ref.targetCount = 1;
      fragmentState.ref.targets = colorTarget;

      final vertexState = arena<WGPUVertexState>();
      vertexState.ref.module = shaderModule;
      vertexState.ref.entryPoint = _createStringView(arena, "vs_main");

      final pipelineDesc = arena<WGPURenderPipelineDescriptor>();
      pipelineDesc.ref.vertex = vertexState.ref;
      pipelineDesc.ref.fragment = fragmentState;
      pipelineDesc.ref.primitive.topology = WGPUPrimitiveTopology.WGPUPrimitiveTopology_TriangleList;
      pipelineDesc.ref.multisample.count = 1;
      pipelineDesc.ref.multisample.mask = 0xFFFFFFFF;

      _renderPipeline = _wgpu.wgpuDeviceCreateRenderPipeline(_device, pipelineDesc);
      _wgpu.wgpuShaderModuleRelease(shaderModule);
    });

    setState(() => _isReady = true);
    _ticker = createTicker((elapsed) => _renderFrame(elapsed))..start();
  }

  void _renderFrame(Duration elapsed) {
    if (!_isReady || _textureHandle == nullptr) return;

    using((arena) {
      WebgpuRend.instance.beginAccessInternal(_textureHandle!);

      const double cycleMs = 4000.0;
      double t = (elapsed.inMilliseconds % cycleMs) / (cycleMs / 2.0);
      double intensity = t > 1.0 ? 2.0 - t : t;

      final encoder = _wgpu.wgpuDeviceCreateCommandEncoder(_device, nullptr);

      // Configure Render Pass
      final colorAttachment = arena<WGPURenderPassColorAttachment>();
      colorAttachment.ref.view = _wgpuTextureView!;
      colorAttachment.ref.loadOp = WGPULoadOp.WGPULoadOp_Clear;
      colorAttachment.ref.storeOp = WGPUStoreOp.WGPUStoreOp_Store;
      colorAttachment.ref.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

      colorAttachment.ref.clearValue.r = intensity * 0.5;
      colorAttachment.ref.clearValue.g = 0.0;
      colorAttachment.ref.clearValue.b = 0.8;
      colorAttachment.ref.clearValue.a = 1.0;

      final passDesc = arena<WGPURenderPassDescriptor>();
      passDesc.ref.colorAttachmentCount = 1;
      passDesc.ref.colorAttachments = colorAttachment;

      final renderPass = _wgpu.wgpuCommandEncoderBeginRenderPass(encoder, passDesc);
      _wgpu.wgpuRenderPassEncoderSetPipeline(renderPass, _renderPipeline!);
      _wgpu.wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
      _wgpu.wgpuRenderPassEncoderEnd(renderPass);

      // Submit Commands
      final commandBuffer = _wgpu.wgpuCommandEncoderFinish(encoder, nullptr);
      final cmdListPtr = arena<Pointer<Void>>();
      cmdListPtr.value = commandBuffer.cast();
      _wgpu.wgpuQueueSubmit(_queue, 1, cmdListPtr.cast());

      _wgpu.wgpuCommandBufferRelease(commandBuffer);
      _wgpu.wgpuCommandEncoderRelease(encoder);

      // Unlock and Present
      WebgpuRend.instance.endAccessInternal(_textureHandle!);
      WebgpuRend.instance.presentInternal(_textureHandle!);
    });
  }

  WGPUStringView _createStringView(Arena arena, String s) {
    final native = s.toNativeUtf8(allocator: arena);
    final view = arena<WGPUStringView>();
    view.ref.data = native.cast();
    view.ref.length = s.length;
    return view.ref;
  }

  @override
  void dispose() {
    _ticker.dispose();
    if (_renderPipeline != null) _wgpu.wgpuRenderPipelineRelease(_renderPipeline!);
    if (_textureHandle != null) WebgpuRend.instance.disposeTextureInternal(_textureHandle!);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isReady) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    
    return Scaffold(
      appBar: AppBar(title: const Text("Raw WebGPU FFI")),
      backgroundColor: Colors.black,
      body: Center(
        child: Container(
          width: _width.toDouble(),
          height: _height.toDouble(),
          decoration: BoxDecoration(
            border: Border.all(color: Colors.white, width: 2),
          ),
          child: Texture(textureId: _textureId),
        ),
      ),
    );
  }
}

```

More examples in the example directory. A simple [gpu_resources.dart](https://github.com/jacksonrl/flutter_webgpu_rend/blob/master/lib/gpu_resources.dart) wrapper also exists but is not stable and should not be relied on unless you are capable of fixing any issues that come up using it. The other examples use this wrapper.


# MacOS, iOS and Linux support

If you want to add these backends, you will need to create a script that downloads the proper dawn binary, and then add some native code that creates a flutter metal texture for iOS/MacOS or an flutter opengl texture on Linux. For iOS this would involve the `FlutterTextureRegistry` Then you will need to hook up that texture to dawn using the dawn API.

# History

This project started as a fork of https://github.com/google/flutter-sw-rend which can only create bitmaps. First I added FFI to that project to reduce latency when uploading to the GPU. Then I wanted to add some shaders, so I created dx11 and OpenGL ES backends, using a shader translation layer from hlsl to glsl. Eventually that became difficult to manage so I switched to WebGPU. Becuase of this, the code quality is varried, as sections got rewritten multiple times. It would probably look different had I started from scratch.
