import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/services.dart' show rootBundle;
import 'package:vector_math/vector_math.dart' as vm;

class MeshGroup {
  final String materialName;
  final int indexStart;
  final int indexCount;
  MeshGroup(this.materialName, this.indexStart, this.indexCount);
}

class MeshData {
  final Float32List vertices;
  final Uint32List indices;
  final List<MeshGroup> groups;
  // The .mtl filename defined in the OBJ
  final String? mtlLibName;

  MeshData(this.vertices, this.indices, this.groups, this.mtlLibName);
}

class ObjLoader {
  static Future<MeshData> load(String assetPath) async {
    final String content = await rootBundle.loadString(assetPath);
    
    final List<vm.Vector3> rawPos = [];
    final List<vm.Vector3> rawNorm = [];
    final Map<String, int> uniqueVertices = {};
    
    final List<double> finalVertexBuffer = [];
    final List<int> finalIndices = [];
    final List<MeshGroup> groups = [];

    String currentMaterial = "default";
    String? mtlLibName;
    int groupStartIndex = 0;

    LineSplitter.split(content).forEach((line) {
      line = line.trim();
      if (line.isEmpty || line.startsWith('#')) return;

      final parts = line.split(RegExp(r'\s+'));
      final type = parts[0];

      if (type == 'v') {
        rawPos.add(vm.Vector3(double.parse(parts[1]), double.parse(parts[2]), double.parse(parts[3])));
      } else if (type == 'vn') {
        rawNorm.add(vm.Vector3(double.parse(parts[1]), double.parse(parts[2]), double.parse(parts[3])));
      } else if (type == 'mtllib') {
        mtlLibName = parts[1];
      } else if (type == 'usemtl') {
        if (finalIndices.length > groupStartIndex) {
          groups.add(MeshGroup(currentMaterial, groupStartIndex, finalIndices.length - groupStartIndex));
        }
        currentMaterial = parts[1];
        groupStartIndex = finalIndices.length;
      } else if (type == 'f') {
        final List<String> faceVerts = parts.sublist(1);
        for (int i = 1; i < faceVerts.length - 1; i++) {
          _processVertex(faceVerts[0], uniqueVertices, finalVertexBuffer, finalIndices, rawPos, rawNorm);
          _processVertex(faceVerts[i], uniqueVertices, finalVertexBuffer, finalIndices, rawPos, rawNorm);
          _processVertex(faceVerts[i + 1], uniqueVertices, finalVertexBuffer, finalIndices, rawPos, rawNorm);
        }
      }
    });

    if (finalIndices.length > groupStartIndex) {
      groups.add(MeshGroup(currentMaterial, groupStartIndex, finalIndices.length - groupStartIndex));
    }

    return MeshData(Float32List.fromList(finalVertexBuffer), Uint32List.fromList(finalIndices), groups, mtlLibName);
  }

  static void _processVertex(String faceToken, Map<String, int> cache, List<double> buffer, List<int> indices, List<vm.Vector3> positions, List<vm.Vector3> normals) {
    if (cache.containsKey(faceToken)) {
      indices.add(cache[faceToken]!);
      return;
    }
    final newIndex = cache.length;
    cache[faceToken] = newIndex;
    indices.add(newIndex);

    final parts = faceToken.split('/');

    //position
    final pIdx = int.parse(parts[0]) - 1;
    final pos = positions[pIdx];
    buffer.add(pos.x); buffer.add(pos.y); buffer.add(pos.z);

    //normal
    if (parts.length > 2 && parts[2].isNotEmpty) {
      final nIdx = int.parse(parts[2]) - 1;
      final norm = normals[nIdx];
      buffer.add(norm.x); buffer.add(norm.y); buffer.add(norm.z);
    } else {
      buffer.add(0.0); buffer.add(1.0); buffer.add(0.0);
    }
  }
}

class MtlLoader {
  /// Loads an MTL file and returns a Map of MaterialName -> Color
  static Future<Map<String, Color>> load(String assetPath) async {
    final String content = await rootBundle.loadString(assetPath);
    final Map<String, Color> materials = {};
    String currentMat = "";

    LineSplitter.split(content).forEach((line) {
      line = line.trim();
      if (line.isEmpty || line.startsWith('#')) return;
      final parts = line.split(RegExp(r'\s+'));
      
      if (parts[0] == 'newmtl') {
        currentMat = parts[1];
      } else if (parts[0] == 'Kd') {
        // Diffuse Color: Kd 0.8 0.8 0.8
        if (currentMat.isNotEmpty && parts.length >= 4) {
          final r = double.parse(parts[1]);
          final g = double.parse(parts[2]);
          final b = double.parse(parts[3]);
          materials[currentMat] = Color.fromARGB(
            255, 
            (r * 255).toInt(), 
            (g * 255).toInt(), 
            (b * 255).toInt()
          );
        }
      }
    });
    return materials;
  }
}