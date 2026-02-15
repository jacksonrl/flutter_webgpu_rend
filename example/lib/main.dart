import 'package:example/image.dart';
import 'package:example/cube.dart';
import 'package:example/object.dart';
import 'package:example/triangle.dart';
import 'package:flutter/material.dart';
import 'package:webgpu_rend/webgpu_rend.dart';


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await WebgpuRend.instance.initialize();
  runApp(const DemoLauncherApp());
}

class DemoLauncherApp extends StatelessWidget {
  const DemoLauncherApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WebGPU Demos',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const MenuScreen(),
    );
  }
}

class MenuScreen extends StatelessWidget {
  const MenuScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Select a Demo')),
      body: ListView(
        children: [
          _buildItem(context, 'Simple Cube', const SimpleCube()),
          _buildItem(context, 'Simple Image', const SimpleImage()),
          _buildItem(context, 'Simple Object', const SimpleObject()),
          _buildItem(context, 'Simple Triangle', const RawWebGpuTriangle()),
        ],
      ),
    );
  }

  Widget _buildItem(BuildContext context, String title, Widget demoWidget) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      child: ListTile(
        title: Text(title),
        trailing: const Icon(Icons.arrow_forward),
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => FloatingBackButtonOverlay(child: demoWidget),
            ),
          );
        },
      ),
    );
  }
}

class FloatingBackButtonOverlay extends StatelessWidget {
  final Widget child;

  const FloatingBackButtonOverlay({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(child: child),
        Positioned(
          top: 40,
          left: 20,
          child: Material(
            color: Colors.transparent,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.5),
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white30, width: 1),
              ),
              child: IconButton(
                icon: const Icon(Icons.arrow_back, color: Colors.white),
                tooltip: "Back to Menu",
                onPressed: () => Navigator.of(context).pop(),
              ),
            ),
          ),
        ),
      ],
    );
  }
}