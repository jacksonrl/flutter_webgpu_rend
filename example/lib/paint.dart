import 'dart:io';

import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:typed_data';
import 'dart:math';

import 'package:webgpu_rend/webgpu_rend.dart';
import 'package:webgpu_rend/gpu_resources.dart';

// --- SHADERS ---

const String kPassthroughVertexWgsl = r'''
    struct VSOutput {
        @builtin(position) position: vec4<f32>,
    };
    @vertex fn main(@builtin(vertex_index) id: u32) -> VSOutput {
        var pos = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -3.0), 
            vec2<f32>(3.0, 1.0), 
            vec2<f32>(-1.0, 1.0)
        );
        var output: VSOutput;
        output.position = vec4<f32>(pos[id], 0.5, 1.0);
        return output;
    }
''';

const String kSimpleBrushFragmentWgsl = r'''
    struct BrushParams { brushCenter: vec2<f32>, brushRadius: f32, brushColor: vec4<f32>, };
    @group(0) @binding(0) var<uniform> u_params: BrushParams;
    @fragment fn main(@builtin(position) screenPos: vec4<f32>) -> @location(0) vec4<f32> {
        let dist = distance(screenPos.xy, u_params.brushCenter);
        if (dist > u_params.brushRadius) { discard; }
        let smoothEdge = 1.0 - smoothstep(u_params.brushRadius - 1.0, u_params.brushRadius, dist);
        return vec4<f32>(u_params.brushColor.rgb, u_params.brushColor.a * smoothEdge);
    }
''';

const String kInstancedBrushVertexWgsl = r'''
    struct Uniforms { brush_color: vec4<f32>, canvas_size: vec2<f32>, brush_size: f32, padding: f32 };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    
    struct VertexInput {
        @location(0) quad_pos: vec2<f32>,
        @location(1) dab_center: vec2<f32>,
    };
    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) dab_center: vec2<f32>,
        @location(1) world_pos: vec2<f32>,
    };
    @vertex fn main(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        let world_pos = in.quad_pos * uniforms.brush_size + in.dab_center;
        let clip_pos = (world_pos / uniforms.canvas_size) * 2.0 - 1.0;
        out.position = vec4(clip_pos.x, -clip_pos.y, 0.5, 1.0);
        out.dab_center = in.dab_center;
        out.world_pos = world_pos;
        return out;
    }
''';

const String kInstancedPenFragmentWgsl = r'''
    struct Uniforms { brush_color: vec4<f32>, canvas_size: vec2<f32>, brush_size: f32, padding: f32 };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) dab_center: vec2<f32>,
        @location(1) world_pos: vec2<f32>,
    };
    @fragment fn main(in: VertexOutput) -> @location(0) vec4<f32> {
        let dist = distance(in.world_pos, in.dab_center); 
        if (dist > uniforms.brush_size) { discard; }
        let falloff = 1.0 - smoothstep(uniforms.brush_size - 1.5, uniforms.brush_size, dist);
        return vec4(uniforms.brush_color.rgb, falloff * uniforms.brush_color.a);
    }
''';

const String kBboxComputeShaderWgsl = r'''
    struct Bbox { data: array<atomic<i32>, 4>, };
    @group(0) @binding(0) var canvasTexture: texture_2d<f32>; 
    @group(0) @binding(1) var<storage, read_write> out_bbox: Bbox;
    
    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let dims = vec2<i32>(textureDimensions(canvasTexture));
        if (global_id.x >= u32(dims.x) || global_id.y >= u32(dims.y)) { return; }
        
        let color = textureLoad(canvasTexture, vec2<i32>(global_id.xy), 0);
        // Check for non-white pixels
        if (color.r < 0.99 || color.g < 0.99 || color.b < 0.99) {
            atomicMin(&out_bbox.data[0], i32(global_id.x));
            atomicMin(&out_bbox.data[1], i32(global_id.y));
            atomicMax(&out_bbox.data[2], i32(global_id.x));
            atomicMax(&out_bbox.data[3], i32(global_id.y));
        }
    }
''';

const String kSimpleComputeWgsl = r'''
    struct Data { elements: array<u32>, };
    @group(0) @binding(0) var<storage, read_write> data: Data;

    @compute @workgroup_size(1, 1, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let i = global_id.x;
        data.elements[i] = data.elements[i] + 1u;
    }
''';

const String kBlurFragmentWgsl = r'''
    struct BlurParams { textureSize: vec2<f32>, direction: vec2<f32>, };
    @group(0) @binding(0) var<uniform> u_params: BlurParams;
    @group(0) @binding(1) var u_texture: texture_2d<f32>;
    @group(0) @binding(2) var u_sampler: sampler;

    struct VSOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, };
    @vertex fn vs_main(@builtin(vertex_index) id: u32) -> VSOutput {
        var pos = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(3.0, 1.0), vec2(-1.0, 1.0));
        var out: VSOutput;
        out.position = vec4(pos[id], 0.0, 1.0);
        out.uv = (pos[id] + 1.0) * 0.5;
        out.uv.y = 1.0 - out.uv.y; 
        return out;
    }

    @fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let texelSize = 1.0 / u_params.textureSize;
        var result = textureSample(u_texture, u_sampler, uv) * 0.227027;
        let off1 = u_params.direction * texelSize * 1.38461538;
        let off2 = u_params.direction * texelSize * 3.23076923;
        result += textureSample(u_texture, u_sampler, uv + off1) * 0.3162162;
        result += textureSample(u_texture, u_sampler, uv - off1) * 0.3162162;
        result += textureSample(u_texture, u_sampler, uv + off2) * 0.070270;
        result += textureSample(u_texture, u_sampler, uv - off2) * 0.070270;
        return result;
    }
''';

const String kDiffFragmentWgsl = r'''
    @group(0) @binding(0) var texSnapshot: texture_2d<f32>;
    @group(0) @binding(1) var texCurrent: texture_2d<f32>;
    @group(0) @binding(2) var samp: sampler;
    
    struct VSOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, };
    @vertex fn vs_main(@builtin(vertex_index) id: u32) -> VSOutput {
        var pos = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(3.0, 1.0), vec2(-1.0, 1.0));
        var out: VSOutput;
        out.position = vec4(pos[id], 0.0, 1.0);
        out.uv = (pos[id] + 1.0) * 0.5;
        out.uv.y = 1.0 - out.uv.y; 
        return out;
    }
    
    @fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let cSnap = textureSample(texSnapshot, samp, uv);
        let cCurr = textureSample(texCurrent, samp, uv);
        
        // If pixels match (distance small), return White (remove them).
        // If pixels differ (distance large), return the Current color (show the new stroke).
        if (distance(cSnap, cCurr) < 0.001) { 
            return vec4(1.0, 1.0, 1.0, 1.0); // White
        } else {
            return cCurr; // Show New Stroke
        }
    }
''';

// --- DATA HELPERS ---

class SimpleBrushUniforms {
  final ByteData _data = ByteData(32);
  SimpleBrushUniforms(Offset center, double radius, Color color) {
    _data.setFloat32(0, center.dx, Endian.host);
    _data.setFloat32(4, center.dy, Endian.host);
    _data.setFloat32(8, radius, Endian.host);
    _data.setFloat32(16, color.red / 255.0, Endian.host);
    _data.setFloat32(20, color.green / 255.0, Endian.host);
    _data.setFloat32(24, color.blue / 255.0, Endian.host);
    _data.setFloat32(28, color.opacity, Endian.host);
  }
  Uint8List get bytes => _data.buffer.asUint8List();
}

class InstancedUniforms {
  final ByteData _data = ByteData(32);
  InstancedUniforms(Size size, double radius, Color color) {
    _data.setFloat32(0, color.red / 255.0, Endian.host);
    _data.setFloat32(4, color.green / 255.0, Endian.host);
    _data.setFloat32(8, color.blue / 255.0, Endian.host);
    _data.setFloat32(12, color.opacity, Endian.host);
    _data.setFloat32(16, size.width, Endian.host);
    _data.setFloat32(20, size.height, Endian.host);
    _data.setFloat32(24, radius, Endian.host);
  }
  Uint8List get bytes => _data.buffer.asUint8List();
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await WebgpuRend.instance.initialize();
  runApp(const SimplePaint());
}

class SimplePaint extends StatelessWidget {
  const SimplePaint({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WebGPU Paint',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const PaintingScreen(),
    );
  }
}

class PaintingScreen extends StatefulWidget {
  const PaintingScreen({super.key});
  @override
  State<PaintingScreen> createState() => _PaintingScreenState();
}

enum BrushType { simple, instanced }

class _PaintingScreenState extends State<PaintingScreen> {
  // --- GPU Resources ---
  GpuTexture? canvasTexture, tempTexture, snapshotTexture;
  GpuSampler? sampler;

  // Pipelines
  GpuRenderPipeline? simplePipeline;
  GpuRenderPipeline? instancedPipeline;
  GpuRenderPipeline? blurPipeline;
  GpuRenderPipeline? diffPipeline;
  GpuComputePipeline? bboxPipeline;
  GpuComputePipeline? simpleComputePipeline;

  // Buffers
  GpuBuffer? simpleUniformBuffer;
  GpuBuffer? instancedUniformBuffer;
  GpuBuffer? blurUniformBuffer;
  GpuBuffer? quadVertexBuffer;
  GpuBuffer? instanceBuffer;
  GpuBuffer? bboxBuffer;
  GpuBuffer? simpleComputeBuffer;

  // Cached BindGroups
  WGPUBindGroup? simpleBindGroup;
  WGPUBindGroup? instancedBindGroup;
  WGPUBindGroup? bboxBindGroup;
  WGPUBindGroup? simpleComputeBindGroup;
  WGPUBindGroup? blurInputCanvasGroup;
  WGPUBindGroup? blurInputTempGroup;
  WGPUBindGroup? diffBindGroup;

  final int _width = 1024;
  final int _height = 1024;

  // --- CPU Resources ---
  Uint8List? _cpuPixelBuffer;
  Rect? _dirtyRect;
  Timer? _drawTimer;

  // --- State & UI ---
  BrushType _brushType = BrushType.simple;
  bool _isGpuPainting = true;
  double _brushSize = 20.0;
  Color _brushColor = Colors.black;
  bool _isLoading = true;
  final TextEditingController _colorController = TextEditingController();

  // Instancing Logic
  final int _maxDabs = 2000;
  final List<double> _dabs = [];
  Offset? _lastPos;

  // UI State
  Rect? _computedBbox;
  double _widgetWidth = 0;
  double _widgetHeight = 0;

  @override
  void initState() {
    super.initState();
    _colorController.text = _colorToHex(_brushColor);
    _initGpu();
  }

  @override
  void dispose() {
    _colorController.dispose();
    _drawTimer?.cancel();
    super.dispose();
  }

  Future<void> _initGpu() async {
    // 1. Textures & Sampler
    canvasTexture = await GpuTexture.create(width: _width, height: _height);
    tempTexture = await GpuTexture.create(width: _width, height: _height);
    snapshotTexture = await GpuTexture.create(width: _width, height: _height);
    sampler = GpuSampler.create();

    // Init CPU buffer with white (255) to match cleared texture
    _cpuPixelBuffer = Uint8List(_width * _height * 4);
    _cpuPixelBuffer!.fillRange(0, _cpuPixelBuffer!.length, 255);

    // 2. Simple Brush
    simplePipeline = GpuRenderPipeline.create(
      blendMode: BlendMode.alpha,
      vertexShader: GpuShader.create(kPassthroughVertexWgsl),
      fragmentShader: GpuShader.create(kSimpleBrushFragmentWgsl),
      bufferLayouts: [],
    );
    simpleUniformBuffer = GpuBuffer.create(
      size: 32,
      usage: WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    );
    simpleBindGroup = simplePipeline!.createBindGroup(0, [
      simpleUniformBuffer!,
    ]);

    // 3. Instanced Brush (Pen)
    instancedPipeline = GpuRenderPipeline.create(
      blendMode: BlendMode.alpha,
      vertexShader: GpuShader.create(kInstancedBrushVertexWgsl),
      fragmentShader: GpuShader.create(kInstancedPenFragmentWgsl),
      bufferLayouts: [
        VertexBufferLayout.fromFormats(
          [WGPUVertexFormat.WGPUVertexFormat_Float32x2], // vec2
          stepMode: WGPUVertexStepMode.WGPUVertexStepMode_Vertex,
          startShaderLocation: 0,
        ),
        VertexBufferLayout.fromFormats(
          [WGPUVertexFormat.WGPUVertexFormat_Float32x2], // vec2
          stepMode: WGPUVertexStepMode.WGPUVertexStepMode_Instance,
          startShaderLocation: 1, 
        ),
      ],
    );

    instancedUniformBuffer = GpuBuffer.create(
      size: 32,
      usage: WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    );
    quadVertexBuffer = GpuBuffer.create(
      size: 32,
      usage: WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
    );
    quadVertexBuffer!.update(
      Float32List.fromList([-1, -1, 1, -1, -1, 1, 1, 1]).buffer.asUint8List(),
    );
    instanceBuffer = GpuBuffer.create(
      size: _maxDabs * 8,
      usage: WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
    );
    instancedBindGroup = instancedPipeline!.createBindGroup(0, [
      instancedUniformBuffer!,
    ]);

    // 4. Blur Effect
    final blurShader = GpuShader.create(kBlurFragmentWgsl);
    blurPipeline = GpuRenderPipeline.create(
      vertexShader: blurShader,
      fragmentShader: blurShader,
      bufferLayouts: [],
      vertexEntryPoint: "vs_main",
      fragmentEntryPoint: "fs_main",
    );
    blurUniformBuffer = GpuBuffer.create(
      size: 16,
      usage: WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    );
    blurInputCanvasGroup = blurPipeline!.createBindGroup(0, [
      blurUniformBuffer!,
      canvasTexture!,
      sampler!,
    ]);
    blurInputTempGroup = blurPipeline!.createBindGroup(0, [
      blurUniformBuffer!,
      tempTexture!,
      sampler!,
    ]);

    // 5. Diff Effect
    final diffShader = GpuShader.create(kDiffFragmentWgsl);
    diffPipeline = GpuRenderPipeline.create(
      vertexShader: diffShader,
      fragmentShader: diffShader,
      bufferLayouts: [],
      vertexEntryPoint: "vs_main",
      fragmentEntryPoint: "fs_main",
    );
    diffBindGroup = diffPipeline!.createBindGroup(0, [
      snapshotTexture!,
      canvasTexture!,
      sampler!,
    ]);

    // 6. BBox Compute
    bboxPipeline = GpuComputePipeline.create(
      GpuShader.create(kBboxComputeShaderWgsl),
    );
    bboxBuffer = GpuBuffer.create(
      size: 16,
      usage:
          WGPUBufferUsage_Storage | 
          WGPUBufferUsage_CopySrc |
          WGPUBufferUsage_CopyDst,
    );
    bboxBindGroup = bboxPipeline!.createBindGroup(0, [
      canvasTexture!,
      bboxBuffer!,
    ]);

    // 7. Simple Compute
    simpleComputePipeline = GpuComputePipeline.create(
      GpuShader.create(kSimpleComputeWgsl)
    );
    simpleComputeBuffer = GpuBuffer.create(
      size: 16, // 4 * uint32
      usage: WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst
    );
    simpleComputeBindGroup = simpleComputePipeline!.createBindGroup(0, [
      simpleComputeBuffer!
    ]);

    _clearCanvas();
    setState(() => _isLoading = false);
  }

  Future<void> _syncCpuBufferFromGpu() async {
    if (canvasTexture == null) return;
    setState(() => _isLoading = true);
    
    try {
      final bytes = await canvasTexture!.download();
      // Ensure local buffer size matches (should be same unless resize logic added later)
      if (_cpuPixelBuffer?.length != bytes.length) {
         _cpuPixelBuffer = bytes;
      } else {
         _cpuPixelBuffer!.setAll(0, bytes);
      }
    } catch (e) {
      debugPrint("Error syncing CPU buffer: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _clearCanvas() {
    if (canvasTexture == null) return;
    setState(() => _computedBbox = null);

    canvasTexture!.beginAccess();
    if (snapshotTexture != null) snapshotTexture!.beginAccess();

    final encoder = CommandEncoder();
    final pass = encoder.beginRenderPass(
      canvasTexture!,
      clearColor: Colors.white,
    );
    pass.end();

    if (snapshotTexture != null) {
      final passSnap = encoder.beginRenderPass(
        snapshotTexture!,
        clearColor: Colors.white,
      );
      passSnap.end();
    }

    final resetData = Int32List.fromList([_width, _height, -1, -1]);
    bboxBuffer!.update(resetData.buffer.asUint8List());

    encoder.submit();

    if (snapshotTexture != null) snapshotTexture!.endAccess();
    canvasTexture!.endAccess();
    canvasTexture!.present();

    // Reset CPU buffer
    _cpuPixelBuffer?.fillRange(0, _cpuPixelBuffer!.length, 255);
  }

  // --- CPU PAINTING LOGIC ---
  void _paintAtCpu(Offset position) {
    if (_cpuPixelBuffer == null) return;
    final int r = _brushSize.round();
    final int centerX = position.dx.round();
    final int centerY = position.dy.round();
    final int left = (centerX - r).clamp(0, _width - 1);
    final int top = (centerY - r).clamp(0, _height - 1);
    final int right = (centerX + r).clamp(0, _width);
    final int bottom = (centerY + r).clamp(0, _height);
    
    final strokeRect = Rect.fromLTRB(
      left.toDouble(),
      top.toDouble(),
      right.toDouble(),
      bottom.toDouble(),
    );
    
    _dirtyRect = _dirtyRect?.expandToInclude(strokeRect) ?? strokeRect;
    
    // Draw circle
    for (int y = top; y < bottom; y++) {
      for (int x = left; x < right; x++) {
        final double dx = x - position.dx;
        final double dy = y - position.dy;
        if (dx * dx + dy * dy <= _brushSize * _brushSize) {
          final int index = (y * _width + x) * 4;
          // Colors: RGBA
          _cpuPixelBuffer![index] = _brushColor.red;
          _cpuPixelBuffer![index + 1] = _brushColor.green;
          _cpuPixelBuffer![index + 2] = _brushColor.blue;
          _cpuPixelBuffer![index + 3] = _brushColor.alpha;
        }
      }
    }
    _scheduleCpuDraw();
  }

  void _scheduleCpuDraw() {
    if (_drawTimer?.isActive ?? false) return;
    _drawTimer = Timer(const Duration(milliseconds: 16), () {
      if (canvasTexture != null && _dirtyRect != null && _cpuPixelBuffer != null) {
        final rect = _dirtyRect!;
        final int w = rect.width.toInt();
        final int h = rect.height.toInt();
        if (w <= 0 || h <= 0) return;

        // Extract the bytes specifically for this rect
        final Uint8List uploadData = Uint8List(w * h * 4);
        
        for (int y = 0; y < h; y++) {
          final int absY = rect.top.toInt() + y;
          final int absX = rect.left.toInt();
          
          final int srcStart = (absY * _width + absX) * 4;
          final int dstStart = y * (w * 4);
          
          final srcEnd = srcStart + (w * 4);
          if (srcEnd <= _cpuPixelBuffer!.length) {
            uploadData.setRange(
              dstStart, 
              dstStart + (w * 4), 
              _cpuPixelBuffer!.sublist(srcStart, srcEnd)
            );
          }
        }

        canvasTexture!.beginAccess();
        // API CALL FIXED AS REQUESTED
        canvasTexture!.uploadRect(uploadData, rect);
        canvasTexture!.endAccess();
        canvasTexture!.present();
        
        _dirtyRect = null;
      }
    });
  }

  // --- GPU PAINTING ---

  void _paintSimple(Offset pos) {
    if (!_isGpuPainting) {
      _paintAtCpu(pos);
      return;
    }

    simpleUniformBuffer!.update(
      SimpleBrushUniforms(pos, _brushSize, _brushColor).bytes,
    );
    canvasTexture!.beginAccess();
    final encoder = CommandEncoder();
    final pass = encoder.beginRenderPass(canvasTexture!);
    pass.bindPipeline(simplePipeline!);
    pass.setBindGroup(0, simpleBindGroup!);
    pass.draw(3);
    pass.end();
    encoder.submit();
    canvasTexture!.endAccess();
    canvasTexture!.present();
  }

  void _paintInstanced(Offset pos) {
    if (_lastPos == null) {
      _lastPos = pos;
      return;
    }
    final dist = (pos - _lastPos!).distance;
    final stepSize = max(1.0, _brushSize * 0.1);
    final steps = (dist / stepSize).ceil();

    for (int i = 0; i <= steps; i++) {
      if (_dabs.length / 2 >= _maxDabs) break;
      final t = steps == 0 ? 0.0 : i / steps;
      final p = Offset.lerp(_lastPos, pos, t)!;
      _dabs.add(p.dx);
      _dabs.add(p.dy);
    }
    _lastPos = pos;

    if (_dabs.isEmpty) return;

    instancedUniformBuffer!.update(
      InstancedUniforms(
        Size(_width.toDouble(), _height.toDouble()),
        _brushSize,
        _brushColor,
      ).bytes,
    );
    instanceBuffer!.update(Float32List.fromList(_dabs).buffer.asUint8List());

    canvasTexture!.beginAccess();
    final encoder = CommandEncoder();
    final pass = encoder.beginRenderPass(canvasTexture!);
    pass.bindPipeline(instancedPipeline!);
    pass.setBindGroup(0, instancedBindGroup!);
    pass.setVertexBuffer(0, quadVertexBuffer!);
    pass.setVertexBuffer(1, instanceBuffer!);
    pass.drawInstanced(4, _dabs.length ~/ 2);
    pass.end();
    encoder.submit();
    canvasTexture!.endAccess();
    canvasTexture!.present();

    _dabs.clear();
  }

  void _onPanStart(Offset pos) {
    _lastPos = null;
    setState(() => _computedBbox = null); 
    if (canvasTexture == null) return;
    
    // SNAPSHOT LOGIC:
    // Copy current state to snapshotTexture BEFORE drawing new stroke.
    // This allows Diff to isolate the new stroke.
    if (snapshotTexture != null) {
        canvasTexture!.beginAccess();
        snapshotTexture!.beginAccess();
        final enc = CommandEncoder();
        enc.copyTextureToTexture(from: canvasTexture!, to: snapshotTexture!);
        enc.submit();
        canvasTexture!.endAccess();
        snapshotTexture!.endAccess();
    }

    if (_brushType == BrushType.simple) {
      _paintSimple(pos);
    } else {
      _paintInstanced(pos);
    }
  }

  void _onPanUpdate(Offset pos) {
    if (canvasTexture == null) return;
    if (_brushType == BrushType.simple) {
      _paintSimple(pos);
    } else {
      _paintInstanced(pos);
    }
  }

  // --- COMPUTE OPERATIONS ---

  Future<void> _runSimpleComputeTest() async {
    if (simpleComputePipeline == null || simpleComputeBuffer == null) return;
    
    final initial = Uint32List.fromList([10, 20, 30, 40]);
    simpleComputeBuffer!.update(initial.buffer.asUint8List());

    final encoder = CommandEncoder();
    final pass = encoder.beginComputePass();
    pass.bindPipeline(simpleComputePipeline!);
    pass.setBindGroup(0, simpleComputeBindGroup!);
    pass.dispatch(initial.length, 1, 1);
    pass.end();
    encoder.submit();

    final resultBytes = await simpleComputeBuffer!.mapRead();
    final resultData = resultBytes.buffer.asUint32List();

    final message = 'Simple Compute Result: $resultData (Should be 11, 21...)';
    if (mounted) {
       ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text(message)));
    }
  }

  Future<void> _runBBox() async {
    if (bboxBuffer == null) return;
    bboxBuffer!.update(
      Int32List.fromList([_width, _height, -1, -1]).buffer.asUint8List(),
    );

    canvasTexture!.beginAccess();
    final encoder = CommandEncoder();
    final pass = encoder.beginComputePass();
    pass.bindPipeline(bboxPipeline!);
    pass.setBindGroup(0, bboxBindGroup!);
    pass.dispatch((_width / 16).ceil(), (_height / 16).ceil());
    pass.end();
    encoder.submit();
    canvasTexture!.endAccess();

    final res = await bboxBuffer!.mapRead();
    final data = res.buffer.asInt32List();

    if (!mounted) return;

    if (data[2] < 0) {
      setState(() => _computedBbox = null);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text("Empty Canvas")));
    } else {
      setState(() {
        _computedBbox = Rect.fromLTRB(
          data[0].toDouble(),
          data[1].toDouble(),
          data[2].toDouble(),
          data[3].toDouble(),
        );
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Bounds: $_computedBbox"))
      );
    }
  }

  // --- EFFECT OPERATIONS ---

  void _runBlur() {
    if (canvasTexture == null) return;
    final hParams = ByteData(16);
    hParams.setFloat32(0, _width.toDouble(), Endian.host);
    hParams.setFloat32(4, _height.toDouble(), Endian.host);
    hParams.setFloat32(8, 1.0, Endian.host);
    hParams.setFloat32(12, 0.0, Endian.host);
    blurUniformBuffer!.update(hParams.buffer.asUint8List());

    canvasTexture!.beginAccess();
    tempTexture!.beginAccess();

    final enc1 = CommandEncoder();
    final pass1 = enc1.beginRenderPass(tempTexture!);
    pass1.bindPipeline(blurPipeline!);
    pass1.setBindGroup(0, blurInputCanvasGroup!);
    pass1.draw(3);
    pass1.end();
    enc1.submit();

    hParams.setFloat32(8, 0.0, Endian.host);
    hParams.setFloat32(12, 1.0, Endian.host);
    blurUniformBuffer!.update(hParams.buffer.asUint8List());

    final enc2 = CommandEncoder();
    final pass2 = enc2.beginRenderPass(canvasTexture!);
    pass2.bindPipeline(blurPipeline!);
    pass2.setBindGroup(0, blurInputTempGroup!);
    pass2.draw(3);
    pass2.end();
    enc2.submit();

    canvasTexture!.endAccess();
    tempTexture!.endAccess();
    canvasTexture!.present();
  }

  void _runDiff() {
    if (canvasTexture == null || snapshotTexture == null) return;

    canvasTexture!.beginAccess();
    snapshotTexture!.beginAccess();
    tempTexture!.beginAccess();

    // 1. Compare Snap vs Canvas -> Write to Temp
    // Shader returns WHITE if matching, CURRENT COLOR if different (the stroke).
    final enc = CommandEncoder();
    final pass = enc.beginRenderPass(tempTexture!);
    pass.bindPipeline(diffPipeline!);
    pass.setBindGroup(0, diffBindGroup!);
    pass.draw(3);
    pass.end();
    enc.submit();

    // 2. Copy Temp -> Canvas (to visualize)
    final enc2 = CommandEncoder();
    enc2.copyTextureToTexture(from: tempTexture!, to: canvasTexture!);
    enc2.submit();

    canvasTexture!.endAccess();
    snapshotTexture!.endAccess();
    tempTexture!.endAccess();
    canvasTexture!.present();
  }

  // --- HELPERS ---

  String _colorToHex(Color color) {
    String toHex(int value) => value.toRadixString(16).padLeft(2, '0');
    return '#${toHex(color.red)}${toHex(color.green)}${toHex(color.blue)}'
        .toUpperCase();
  }

  Color? _colorFromHex(String hex) {
    final hexCode = hex.toUpperCase().replaceAll('#', '');
    if (hexCode.length == 6) {
      try {
        final int value = int.parse('FF$hexCode', radix: 16);
        return Color(value);
      } catch (e) {
        return null;
      }
    }
    return null;
  }

  Offset _mapPos(Offset local) {
    final scale = min(_widgetWidth / _width, _widgetHeight / _height);
    final offsetX = (_widgetWidth - _width * scale) / 2;
    final offsetY = (_widgetHeight - _height * scale) / 2;
    return Offset((local.dx - offsetX) / scale, (local.dy - offsetY) / scale);
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: CircularProgressIndicator());
    final isInstanced = _brushType == BrushType.instanced;

    return Scaffold(
      appBar: AppBar(
        title: Text("WebGPU Paint ${_isGpuPainting ? '(GPU)' : '(CPU)'}"),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.calculate),
            onPressed: _runSimpleComputeTest,
            tooltip: "Simple Compute Test",
          ),
          IconButton(
            icon: const Icon(Icons.difference),
            onPressed: _runDiff,
            tooltip: "Show Diff (Last Stroke)",
          ),
          IconButton(
            icon: const Icon(Icons.blur_on),
            onPressed: _runBlur,
            tooltip: "Blur",
          ),
          IconButton(
            icon: const Icon(Icons.crop_free),
            onPressed: _runBBox,
            tooltip: "Calc Bounds",
          ),
          IconButton(
            icon: const Icon(Icons.delete),
            onPressed: _clearCanvas,
            tooltip: "Clear",
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            color: Colors.grey[900],
            child: Column(
              children: [
                Row(
                  children: [
                    const Text("Mode: "),
                    DropdownButton<BrushType>(
                      value: _brushType,
                      dropdownColor: Colors.grey[800],
                      items: const [
                        DropdownMenuItem(
                          value: BrushType.simple,
                          child: Text("Simple"),
                        ),
                        DropdownMenuItem(
                          value: BrushType.instanced,
                          child: Text("Instanced Pen"),
                        ),
                      ],
                      onChanged: (v) {
                         setState(() { 
                           _brushType = v!;
                           if(_brushType == BrushType.instanced) _isGpuPainting = true;
                         });
                      },
                    ),
                    const SizedBox(width: 16),
                    const Text("GPU: "),
                    Switch(
                      value: _isGpuPainting,
                      onChanged: isInstanced ? null : (v) async {
                         setState(() => _isGpuPainting = v);
                         if (!v) {
                           await _syncCpuBufferFromGpu();
                         }
                      },
                    ),
                    const SizedBox(width: 16),
                    // Hex Color Input
                    SizedBox(
                      width: 100,
                      child: SizedBox(
                        height: 40,
                        child: TextField(
                          controller: _colorController,
                          style: const TextStyle(fontSize: 14),
                          decoration: const InputDecoration(
                            labelText: "Hex (#RRGGBB)",
                            border: OutlineInputBorder(),
                            contentPadding: EdgeInsets.symmetric(horizontal: 8),
                          ),
                          onSubmitted: (val) {
                            final c = _colorFromHex(val);
                            if (c != null) {
                              setState(() => _brushColor = c);
                            } else {
                               ScaffoldMessenger.of(context).showSnackBar(
                                 const SnackBar(content: Text("Invalid Hex"))
                               );
                            }
                          },
                        ),
                      ),
                    ),
                    Platform.isWindows ?
                    Expanded(
                      child: Row(
                        children: [
                          const Text("Size: "),
                          Expanded(
                            child: Slider(
                              min: 1.0,
                              max: 150.0,
                              value: _brushSize,
                              onChanged: (v) => setState(() => _brushSize = v),
                            ),
                          ),
                          Text("${_brushSize.toInt()}px"),
                        ],
                      ),
                    )
                    :SizedBox(),
                  ],
                ),
                const SizedBox(height: 8),
                
              ],
            ),
          ),
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                _widgetWidth = constraints.maxWidth;
                _widgetHeight = constraints.maxHeight;

                Rect? displayBbox;
                if (_computedBbox != null) {
                  final scale = min(
                    _widgetWidth / _width,
                    _widgetHeight / _height,
                  );
                  final offsetX = (_widgetWidth - _width * scale) / 2;
                  final offsetY = (_widgetHeight - _height * scale) / 2;

                  displayBbox = Rect.fromLTRB(
                    _computedBbox!.left * scale + offsetX,
                    _computedBbox!.top * scale + offsetY,
                    _computedBbox!.right * scale + offsetX,
                    _computedBbox!.bottom * scale + offsetY,
                  );
                }

                return GestureDetector(
                  onPanStart: (d) => _onPanStart(_mapPos(d.localPosition)),
                  onPanUpdate: (d) => _onPanUpdate(_mapPos(d.localPosition)),
                  onPanEnd: (d) => _lastPos = null,
                  child: Container(
                    color: Colors.grey[200],
                    child: Stack(
                      children: [
                        Center(
                          child: AspectRatio(
                            aspectRatio: _width / _height,
                            child: Container(
                              decoration: BoxDecoration(
                                border: Border.all(color: Colors.black),
                                boxShadow: const [
                                  BoxShadow(
                                    blurRadius: 10,
                                    color: Colors.black26,
                                  ),
                                ],
                              ),
                              child: Texture(
                                textureId: canvasTexture!.textureId,
                                filterQuality: FilterQuality.medium,
                              ),
                            ),
                          ),
                        ),
                        if (displayBbox != null)
                          Positioned.fromRect(
                            rect: displayBbox,
                            child: Container(
                              decoration: BoxDecoration(
                                border: Border.all(color: Colors.red, width: 3),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          Platform.isAndroid?
          SafeArea(
            child: Row(
              children: [
                const Text("Size: "),
                Expanded(
                  child: Slider(
                    min: 1.0,
                    max: 150.0,
                    value: _brushSize,
                    onChanged: (v) => setState(() => _brushSize = v),
                  ),
                ),
                Text("${_brushSize.toInt()}px"),
              ],
            ),
          ): SizedBox(),         
        ],
      ),
    );
  }
}