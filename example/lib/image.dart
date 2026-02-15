import 'dart:ffi';
import 'package:flutter/material.dart';
import 'package:webgpu_rend/webgpu_rend.dart';
import 'package:webgpu_rend/gpu_resources.dart';
import 'package:ffi/ffi.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    await WebgpuRend.instance.initialize();
  } catch (e) {
    print("Init Error: $e");
    return;
  }
  runApp(const MyApp());
}

class SimpleImage extends StatelessWidget {
  const SimpleImage({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyApp(),
    );
  }
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  GpuTexture? texture;
  bool _running = true;

  @override
  void initState() {
    super.initState();
    _init();
  }

  @override
  void dispose() {
    _running = false;
    //releasing causes problems for some reason
    //texture?.dispose();
    super.dispose();
  }

  Future<void> _init() async {
    final newTexture = await GpuTexture.create(width: 500, height: 500);

    if (!mounted) {
      newTexture.dispose();
      return;
    }

    texture = newTexture;
    
    setState(() {});
    _renderLoop();
  }

  void _renderLoop() async {
    final wgpu = WebgpuRend.instance.wgpu;
    final device = WebgpuRend.instance.device;
    final queue = WebgpuRend.instance.queue;

    int frame = 0;
    const int speed = 100;

    while (_running && mounted) {
      await Future.delayed(const Duration(milliseconds: 16));

      if (!_running || texture == null) break;

      texture!.beginAccess();

      final encoder = wgpu.wgpuDeviceCreateCommandEncoder(device, nullptr);

      using((arena) {
        final colorAttr = arena<WGPURenderPassColorAttachment>();

        colorAttr.ref.view = texture!.view;
        colorAttr.ref.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

        colorAttr.ref.loadOp = WGPULoadOp.WGPULoadOp_Clear;
        colorAttr.ref.storeOp = WGPUStoreOp.WGPUStoreOp_Store;

        double t = (frame % (speed * 2)) / speed.toDouble();

        double r = t > 1.0 ? 2.0 - t : t;

        colorAttr.ref.clearValue.r = r;
        colorAttr.ref.clearValue.g = 0.0;
        colorAttr.ref.clearValue.b = 1.0;
        colorAttr.ref.clearValue.a = 1.0;
        colorAttr.ref.resolveTarget = nullptr;

        final passDesc = arena<WGPURenderPassDescriptor>();
        passDesc.ref.colorAttachmentCount = 1;
        passDesc.ref.colorAttachments = colorAttr;
        passDesc.ref.depthStencilAttachment = nullptr;
        passDesc.ref.timestampWrites = nullptr;
        passDesc.ref.occlusionQuerySet = nullptr;

        final pass = wgpu.wgpuCommandEncoderBeginRenderPass(encoder, passDesc);
        wgpu.wgpuRenderPassEncoderEnd(pass);
      });

      final cmdBuf = wgpu.wgpuCommandEncoderFinish(encoder, nullptr);

      using((arena) {
        final qPtr = arena<Pointer<Void>>();
        qPtr.value = cmdBuf.cast();
        wgpu.wgpuQueueSubmit(queue, 1, qPtr.cast());
      });

      wgpu.wgpuCommandBufferRelease(cmdBuf);

      texture!.endAccess();

      texture!.present();
      frame++;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (texture == null) {
      return const Center(child: CircularProgressIndicator());
    }
    return Scaffold(
      body: Center(
        child: SizedBox(
          width: 500,
          height: 500,
          child: Texture(textureId: texture!.textureId),
        ),
      ),
    );
  }
}
