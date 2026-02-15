import 'dart:async';
import 'dart:ffi';
import 'dart:math';

import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:vector_math/vector_math.dart' as vm;
import 'package:webgpu_rend/webgpu_rend.dart';
import 'package:webgpu_rend/gpu_resources.dart';
import 'package:webgpu_rend/obj_load.dart';

// Simple Diffuse Lighting
const String kLitShader = r'''
struct Uniforms { 
    mvp : mat4x4<f32>,
    color : vec4<f32>,
};
@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
};
struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) vNormal : vec3<f32>,
};

@vertex fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.Position = u.mvp * vec4<f32>(in.position, 1.0);
    out.vNormal = in.normal;
    return out;
}

@fragment fn fs_main(@location(0) vNormal : vec3<f32>) -> @location(0) vec4<f32> {
    // Simple Directional Light from top-right-front
    let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    
    // Diffuse calculation
    let diff = max(dot(normalize(vNormal), lightDir), 0.0);
    
    // Ambient light
    let ambient = 0.3;
    
    let lighting = diff + ambient;
    
    return vec4<f32>(u.color.rgb * lighting, 1.0);
}
''';

class DrawUniforms {
  final Pointer<Uint8> _ptr = calloc<Uint8>(80); // 64 (Mat4) + 16 (Color)
  
  void update(vm.Matrix4 mvp, Color color) {
    final floatList = _ptr.cast<Float>();
    
    for (int i = 0; i < 16; i++) {
      floatList[i] = mvp.storage[i];
    }
    
    floatList[16] = color.red / 255.0;
    floatList[17] = color.green / 255.0;
    floatList[18] = color.blue / 255.0;
    floatList[19] = 1.0;
  }
  
  Pointer<Void> get ptr => _ptr.cast();
  int get size => 80;
  void dispose() => calloc.free(_ptr);
}

class OrbitCamera {
  double theta = 0.0; 
  double phi = pi / 2; 
  double radius = 15.0;

  void handlePan(double dx, double dy) {
    theta -= dx * 0.01;
    phi -= dy * 0.01;
    phi = phi.clamp(0.1, pi - 0.1); 
  }
  void handleZoom(double scale) {
    radius /= scale;
    radius = radius.clamp(2.0, 100.0);
  }
  vm.Matrix4 getViewMatrix() {
    final x = radius * sin(phi) * sin(theta);
    final y = radius * cos(phi);
    final z = radius * sin(phi) * cos(theta);
    return vm.makeViewMatrix(vm.Vector3(x, y, z), vm.Vector3.zero(), vm.Vector3(0, 1, 0));
  }
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await WebgpuRend.instance.initialize();
  runApp(SimpleObject());
}

class SimpleObject extends StatelessWidget {
  const SimpleObject({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(home: SwordViewer());
  }
}

class SwordViewer extends StatefulWidget {
  const SwordViewer({super.key});
  @override
  State<SwordViewer> createState() => _SwordViewerState();
}

class _SwordViewerState extends State<SwordViewer> with SingleTickerProviderStateMixin {
  // GPU Resources
  GpuTexture? canvasTex, msaaTex, depthTex;
  GpuRenderPipeline? pipeline;
  GpuBuffer? vertexBuffer, indexBuffer;
  
  List<GpuBuffer> uniformBuffers = [];
  List<WGPUBindGroup> bindGroups = [];
  List<DrawUniforms> cpuUniforms = [];

  MeshData? _mesh;
  Map<String, Color> _materials = {};
  
  final OrbitCamera _cam = OrbitCamera();
  late Ticker _ticker;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    const w = 800, h = 600;
    canvasTex = await GpuTexture.create(width: w, height: h);
    msaaTex = GpuTexture.createMsaa(width: w, height: h, samples: 4);
    depthTex = GpuTexture.createDepth(width: w, height: h, samples: 4);

    _mesh = await ObjLoader.load('assets/sword.obj');

    await _loadMaterials();

    // Pipeline Setup
    final shader = GpuShader.create(kLitShader);
    pipeline = GpuRenderPipeline.create(
      vertexShader: shader,
      fragmentShader: shader,
      vertexEntryPoint: "vs_main",
      fragmentEntryPoint: "fs_main",
      enableDepth: true,
      sampleCount: 4,
      topology: WGPUPrimitiveTopology.WGPUPrimitiveTopology_TriangleList,
      cullMode: WGPUCullMode.WGPUCullMode_Back,
      bufferLayouts: [
        VertexBufferLayout.fromFormats([
          WGPUVertexFormat.WGPUVertexFormat_Float32x3,
          WGPUVertexFormat.WGPUVertexFormat_Float32x3,
        ])
      ],
    );

    // Geometry Buffers
    vertexBuffer = GpuBuffer.create(
      size: _mesh!.vertices.lengthInBytes,
      usage: WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
    );
    vertexBuffer!.update(_mesh!.vertices.buffer.asUint8List());

    indexBuffer = GpuBuffer.create(
      size: _mesh!.indices.lengthInBytes,
      usage: WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
    );
    indexBuffer!.update(_mesh!.indices.buffer.asUint8List());

    // Create Uniforms
    for (var _ in _mesh!.groups) {
      final uBuf = GpuBuffer.create(
        size: 256, 
        usage: WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      );
      uniformBuffers.add(uBuf);
      cpuUniforms.add(DrawUniforms());
      bindGroups.add(pipeline!.createBindGroup(0, [uBuf]));
    }

    setState(() => _isLoading = false);
    _ticker = createTicker((_) => _render())..start();
  }

  Future<void> _loadMaterials() async {
    final file = 'assets/sword.mtl';
    try {
      _materials = await MtlLoader.load(file);
    } catch (e) {
      print("Error loading MTL: $e");
      _materials = {};
    }
  }

  void _render() {
    if (canvasTex == null || _mesh == null) return;

    final proj = vm.makePerspectiveMatrix(vm.radians(45), 800/600, 0.1, 100.0);
    final view = _cam.getViewMatrix();
    final model = vm.Matrix4.identity()
      ..rotateX(pi)
      ..rotateY(pi/2);
    //final model = vm.Matrix4.identity()..translate(10.0, -5.0, 0.0);
    final mvp = proj * view * model;

    canvasTex!.beginAccess();
    final encoder = CommandEncoder();
    final pass = encoder.beginRenderPass(
      canvasTex!,
      msaaTexture: msaaTex,
      depthTexture: depthTex,
      sampleCount: 4,
      clearColor: const Color(0xFF101010),
    );

    pass.bindPipeline(pipeline!);
    pass.setVertexBuffer(0, vertexBuffer!);
    pass.setIndexBuffer(indexBuffer!, WGPUIndexFormat.WGPUIndexFormat_Uint32);

    for (int i = 0; i < _mesh!.groups.length; i++) {
      final group = _mesh!.groups[i];
      
      // Look up color in our map, or default to pink (so we know it's missing)
      final color = _materials[group.materialName] ?? Colors.pinkAccent;
      
      cpuUniforms[i].update(mvp, color);
      uniformBuffers[i].updateRaw(cpuUniforms[i].ptr, cpuUniforms[i].size);

      pass.setBindGroup(0, bindGroups[i]);
      pass.drawIndexed(group.indexCount, 1, group.indexStart);
    }

    pass.end();
    encoder.submit();
    canvasTex!.endAccess();
    canvasTex!.present();
  }

  @override
  void dispose() {
    _ticker.dispose();
    for (var u in cpuUniforms) {
      u.dispose();
    }
    
    /*
    disposing causes issues for some reason when re-entering..
    vertexBuffer?.dispose();
    indexBuffer?.dispose();
    for(var b in uniformBuffers) {
      b.dispose();
    }
    
    canvasTex?.dispose();
    msaaTex?.dispose();
    depthTex?.dispose();
    */
    
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: CircularProgressIndicator());
    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        onScaleUpdate: (d) {
          _cam.handlePan(d.focalPointDelta.dx, d.focalPointDelta.dy);
          if (d.pointerCount > 1) _cam.handleZoom(d.scale);
        },
        child: Center(
          child: Container(
            width: 800,
            height: 600,
            decoration: BoxDecoration(border: Border.all(color: Colors.white24)),
            child: Texture(textureId: canvasTex!.textureId),
          ),
        ),
      ),
    );
  }
}
