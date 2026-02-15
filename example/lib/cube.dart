import 'dart:async';
import 'dart:developer';
import 'dart:ffi';
import 'dart:math';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:vector_math/vector_math.dart' as vm;
import 'package:webgpu_rend/webgpu_rend.dart';
import 'package:webgpu_rend/gpu_resources.dart';

const String kCubeShaderWgsl = r'''
struct Uniforms {
    modelViewProjectionMatrix : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) color : vec3<f32>,
};

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragColor : vec3<f32>,
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.Position = uniforms.modelViewProjectionMatrix * vec4<f32>(input.position, 1.0);
    output.fragColor = input.color;
    return output;
}

@fragment
fn fs_main(@location(0) fragColor : vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(fragColor, 1.0);
}
''';

Float32List cubeVertexData() {
  final List<double> vertices = [
    -1.0, -1.0, 1.0, 1.0, 0.0, 0.0,
    1.0, -1.0, 1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    -1.0, 1.0, 1.0, 1.0, 0.0, 0.0,

    1.0, -1.0, -1.0, 0.0, 1.0, 0.0,
    -1.0, -1.0, -1.0, 0.0, 1.0, 0.0,
    -1.0, 1.0, -1.0, 0.0, 1.0, 0.0,
    1.0, 1.0, -1.0, 0.0, 1.0, 0.0,

    -1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
    1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
    1.0, 1.0, -1.0, 0.0, 0.0, 1.0,
    -1.0, 1.0, -1.0, 0.0, 0.0, 1.0,

    -1.0, -1.0, -1.0, 1.0, 1.0, 0.0,
    1.0, -1.0, -1.0, 1.0, 1.0, 0.0,
    1.0, -1.0, 1.0, 1.0, 1.0, 0.0,
    -1.0, -1.0, 1.0, 1.0, 1.0, 0.0,

    1.0, -1.0, 1.0, 1.0, 0.0, 1.0,
    1.0, -1.0, -1.0, 1.0, 0.0, 1.0,
    1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 0.0, 1.0,

    -1.0, -1.0, -1.0, 0.0, 1.0, 1.0,
    -1.0, -1.0, 1.0, 0.0, 1.0, 1.0,
    -1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    -1.0, 1.0, -1.0, 0.0, 1.0, 1.0,
  ];
  return Float32List.fromList(vertices);
}

Uint16List cubeIndexData() {
  final List<int> indices = [];
  for (int i = 0; i < 6; i++) {
    int base = i * 4;
    indices.addAll([
      base + 0, base + 1, base + 2,
      base + 0, base + 2, base + 3 
    ]);
  }
  return Uint16List.fromList(indices);
}

class NativeCameraUniforms {
  final Pointer<Float> _ptr;
  NativeCameraUniforms() : _ptr = calloc<Float>(16);
  void update(vm.Matrix4 matrix) {
    for (int i = 0; i < 16; i++) {
      _ptr[i] = matrix.storage[i];
    }
  }
  Pointer<Void> get ptr => _ptr.cast();
  int get size => 16 * 4;
  void dispose() => calloc.free(_ptr);
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await WebgpuRend.instance.initialize();
  runApp(const SimpleCube());
}

class SimpleCube extends StatelessWidget {
  const SimpleCube({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WebGPU Index draw',
      theme: ThemeData.dark(),
      home: const CubeScreen(),
    );
  }
}

class CubeScreen extends StatefulWidget {
  const CubeScreen({super.key});
  @override
  State<CubeScreen> createState() => _CubeScreenState();
}

class _CubeScreenState extends State<CubeScreen>
    with SingleTickerProviderStateMixin {
  GpuTexture? canvasTexture;
  GpuTexture? depthTexture;

  GpuRenderPipeline? pipeline;
  GpuBuffer? vertexBuffer;
  GpuBuffer? indexBuffer;
  GpuBuffer? uniformBuffer;
  WGPUBindGroup? bindGroup;

  late NativeCameraUniforms _cameraUniforms;
  late Ticker _ticker;
  double _time = 0.0;

  final int _displayW = 800;
  final int _displayH = 600;

  bool _isLoading = true;
  double _fps = 0.0;
  final List<double> _frameTimes = [];
  final Stopwatch _frameWatch = Stopwatch();

  @override
  void initState() {
    super.initState();
    _cameraUniforms = NativeCameraUniforms();
    _initGpu();
  }

  Future<void> _initGpu() async {
    canvasTexture = await GpuTexture.create(width: _displayW, height: _displayH);

    depthTexture = GpuTexture.createDepth(
      width: _displayW, 
      height: _displayH,
    );

    final shader = GpuShader.create(kCubeShaderWgsl);
    pipeline = GpuRenderPipeline.create(
      vertexShader: shader,
      fragmentShader: shader,
      vertexEntryPoint: "vs_main",
      fragmentEntryPoint: "fs_main",
      enableDepth: true,
      sampleCount: 1, 
      topology: WGPUPrimitiveTopology.WGPUPrimitiveTopology_TriangleList,
      cullMode: WGPUCullMode.WGPUCullMode_Back,
      frontFace: WGPUFrontFace.WGPUFrontFace_CCW,
      bufferLayouts: [
        VertexBufferLayout.fromFormats(
          [
            WGPUVertexFormat.WGPUVertexFormat_Float32x3,
            WGPUVertexFormat.WGPUVertexFormat_Float32x3,
          ],
          stepMode: WGPUVertexStepMode.WGPUVertexStepMode_Vertex,
        ),
      ],
    );

    final vData = cubeVertexData();
    vertexBuffer = GpuBuffer.create(
      size: vData.lengthInBytes,
      usage: WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
    );
    vertexBuffer!.update(vData.buffer.asUint8List());

    final iData = cubeIndexData();
    indexBuffer = GpuBuffer.create(
      size: iData.lengthInBytes,
      usage: WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
    );
    indexBuffer!.update(iData.buffer.asUint8List());

    uniformBuffer = GpuBuffer.create(
      size: 64,
      usage: WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    );
    bindGroup = pipeline!.createBindGroup(0, [uniformBuffer!]);

    setState(() => _isLoading = false);
    _ticker = createTicker(_onTick)..start();
  }

  void _onTick(Duration elapsed) {
    if (canvasTexture == null) return;
    if (_frameWatch.isRunning) {
      _frameTimes.add(1000.0 / _frameWatch.elapsedMilliseconds);
      if (_frameTimes.length > 60) _frameTimes.removeAt(0);
      if (_frameTimes.length > 5) {
        _fps = _frameTimes.reduce((a, b) => a + b) / _frameTimes.length;
      }
      _frameWatch.reset();
    }
    _frameWatch.start();
    if (mounted && _frameTimes.length % 30 == 0) setState(() {});

    _time = elapsed.inMilliseconds / 1000.0;
    _render();
  }

  void _render() {
    Timeline.startSync('WebGPU Render');

    final aspect = _displayW / _displayH;
    final projection = vm.makePerspectiveMatrix(vm.radians(45), aspect, 0.1, 100.0);
    final view = vm.makeViewMatrix(
      vm.Vector3(sin(_time) * 5, 2, cos(_time) * 5),
      vm.Vector3(0, 0, 0),
      vm.Vector3(0, 1, 0),
    );
    final model = vm.Matrix4.rotationX(_time * 0.5) * vm.Matrix4.rotationZ(_time * 0.3);
    _cameraUniforms.update(projection * view * model);
    uniformBuffer!.updateRaw(_cameraUniforms.ptr, _cameraUniforms.size);

    canvasTexture!.beginAccess();

    final encoder = CommandEncoder();
    
    final pass = encoder.beginRenderPass(
      canvasTexture!,
      depthTexture: depthTexture,
      clearColor: Colors.black,
    );

    pass.bindPipeline(pipeline!);
    pass.setBindGroup(0, bindGroup!);
    
    pass.setVertexBuffer(0, vertexBuffer!);
    
    pass.setIndexBuffer(indexBuffer!, WGPUIndexFormat.WGPUIndexFormat_Uint16);
    
    pass.drawIndexed(36);
    
    pass.end();

    encoder.submit();
    canvasTexture!.endAccess();
    canvasTexture!.present();

    Timeline.finishSync();
  }

  @override
  void dispose() {
    _ticker.dispose();
    _cameraUniforms.dispose();

    /*
    //disposing causes issues when going back, maybe becuase
    //flutter keeps trying to draw the the texture while the animation
    //slides out?
    canvasTexture?.dispose();
    depthTexture?.dispose();
    vertexBuffer?.dispose();
    indexBuffer?.dispose();
    uniformBuffer?.dispose();
    pipeline?.dispose();
    */
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: CircularProgressIndicator());
    return Scaffold(
      backgroundColor: Colors.black87,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("FPS: ${_fps.toStringAsFixed(1)}", style: const TextStyle(color: Colors.greenAccent, fontSize: 20)),
            const SizedBox(height: 10),
            Container(
              width: _displayW.toDouble(),
              height: _displayH.toDouble(),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.white24),
              ),
              child: Texture(
                textureId: canvasTexture!.textureId,
                filterQuality: FilterQuality.medium,
              ),
            ),
            const SizedBox(height: 10),
            const Text("Indexed Drawing", style: TextStyle(color: Colors.grey)),
          ],
        ),
      ),
    );
  }
}