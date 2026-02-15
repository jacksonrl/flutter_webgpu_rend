import 'dart:async';
import 'dart:ffi';
import 'dart:typed_data';
import 'dart:io';
import 'dart:ui';
import 'package:ffi/ffi.dart';
import 'package:vector_math/vector_math.dart';
import 'package:webgpu_rend/webgpu_rend.dart';

WGPUTextureFormat get kPreferredTextureFormat => Platform.isAndroid
    ? WGPUTextureFormat.WGPUTextureFormat_RGBA8Unorm
    : WGPUTextureFormat.WGPUTextureFormat_BGRA8Unorm;

// Consider removing this in favor of arena
class _Scratchpad {
  static final _Scratchpad instance = _Scratchpad._();
  late final Pointer<WGPURenderPassDescriptor> renderPassDesc;
  late final Pointer<WGPURenderPassColorAttachment> colorAttachment;
  late final Pointer<WGPURenderPassDepthStencilAttachment>
      depthStencilAttachment;

  _Scratchpad._() {
    renderPassDesc = calloc<WGPURenderPassDescriptor>();
    colorAttachment = calloc<WGPURenderPassColorAttachment>();
    depthStencilAttachment = calloc<WGPURenderPassDepthStencilAttachment>();
  }
}

class VertexAttribute {
  final WGPUVertexFormat format;
  final int offset;
  final int shaderLocation;

  const VertexAttribute({
    required this.format,
    required this.offset,
    required this.shaderLocation,
  });
}

class VertexBufferLayout {
  final int arrayStride;
  final WGPUVertexStepMode stepMode;
  final List<VertexAttribute> attributes;
  VertexBufferLayout({
    required this.arrayStride,
    required this.stepMode,
    required this.attributes,
  });

  static VertexBufferLayout fromFormats(
    List<WGPUVertexFormat> formats, {
    WGPUVertexStepMode stepMode = WGPUVertexStepMode.WGPUVertexStepMode_Vertex,
    int startShaderLocation = 0,
  }) {
    final List<VertexAttribute> attrs = [];
    int currentOffset = 0;

    for (int i = 0; i < formats.length; i++) {
      final format = formats[i];
      attrs.add(VertexAttribute(
        format: format,
        offset: currentOffset,
        shaderLocation: startShaderLocation + i,
      ));
      currentOffset += _getSizeInBytes(format);
    }

    return VertexBufferLayout(
      arrayStride: currentOffset, // Total size is the stride
      stepMode: stepMode,
      attributes: attrs,
    );
  }

  static int _getSizeInBytes(WGPUVertexFormat format) {
    switch (format) {
      case WGPUVertexFormat.WGPUVertexFormat_Float32: return 4;
      case WGPUVertexFormat.WGPUVertexFormat_Float32x2: return 8;
      case WGPUVertexFormat.WGPUVertexFormat_Float32x3: return 12;
      case WGPUVertexFormat.WGPUVertexFormat_Float32x4: return 16;
      default: throw UnimplementedError("Size not defined for format: $format");
    }
  }
}

WGPUStringView _createStringView(Arena arena, String s) {
  final nativeStr = s.toNativeUtf8(allocator: arena);
  final view = arena<WGPUStringView>();
  view.ref.data = nativeStr.cast();
  view.ref.length = s.length;
  return view.ref;
}

WGPUBindGroup _createBindGroupHelper(
    WGPUBindGroupLayout layout, List<Object> resources) {
  final wgpu = WebgpuRend.instance.wgpu;
  return using((arena) {
    final bgEntries = arena<WGPUBindGroupEntry>(resources.length);
    for (int i = 0; i < resources.length; i++) {
      final r = resources[i];
      final bgEntry = bgEntries.elementAt(i);
      bgEntry.ref.binding = i;
      bgEntry.ref.buffer = nullptr;
      bgEntry.ref.textureView = nullptr;
      bgEntry.ref.sampler = nullptr;
      if (r is GpuBuffer) {
        bgEntry.ref.buffer = r.handle.cast();
        bgEntry.ref.size = r.size;
        bgEntry.ref.offset = 0;
      } else if (r is WGPUTextureView) {
        bgEntry.ref.textureView = r;
      } else if (r is GpuTexture) {
        bgEntry.ref.textureView = r.view;
      } else if (r is GpuSampler) {
        bgEntry.ref.sampler = r.handle.cast();
      } else {
        throw "Unsupported resource type: $r";
      }
    }
    final groupDesc = arena<WGPUBindGroupDescriptor>();
    groupDesc.ref.label.data = nullptr;
    groupDesc.ref.label.length = 0;
    groupDesc.ref.layout = layout;
    groupDesc.ref.entryCount = resources.length;
    groupDesc.ref.entries = bgEntries;
    return wgpu.wgpuDeviceCreateBindGroup(
        WebgpuRend.instance.device, groupDesc);
  });
}

class GpuResource {
  final Pointer<Void> handle;
  GpuResource(this.handle);
}

final _textureFinalizer =
    NativeFinalizer(WebgpuRend.instance.disposeTexturePtr);

class GpuTexture implements Finalizable {
  final Pointer<Void> _handle;
  final int textureId;
  final WGPUTexture texture;
  final WGPUTextureView view;
  final int width;
  final int height;
  final bool _isShared;

  bool _disposed = false;

  GpuTexture._(this._handle, this.textureId, this.texture, this.view,
      this.width, this.height, this._isShared) {
    if (_isShared) {
      _textureFinalizer.attach(this, _handle.cast(), detach: this);
    }
  }

  static Future<GpuTexture> create(
      {required int width, required int height}) async {
    final sw = WebgpuRend.instance;
    final handle = sw.createTextureInternal(width, height);
    if (handle == nullptr) throw "Failed to create texture";

    final id = sw.getTextureIdInternal(handle);
    final rawTexPtr = sw.getWgpuTextureInternal(handle);
    final rawViewPtr = sw.getWgpuViewInternal(handle);

    return GpuTexture._(
        handle, id, rawTexPtr.cast(), rawViewPtr.cast(), width, height, true);
  }

  static GpuTexture createDepth({required int width, required int height, int samples = 1}) {
    final wgpu = WebgpuRend.instance.wgpu;
    return using((arena) {
      final desc = arena<WGPUTextureDescriptor>();
      desc.ref.label.data = nullptr;
      desc.ref.label.length = 0;
      desc.ref.usage = WGPUTextureUsage_RenderAttachment;
      desc.ref.dimension = WGPUTextureDimension.WGPUTextureDimension_2D;
      desc.ref.size.width = width;
      desc.ref.size.height = height;
      desc.ref.size.depthOrArrayLayers = 1;
      desc.ref.format = WGPUTextureFormat.WGPUTextureFormat_Depth24Plus;
      desc.ref.mipLevelCount = 1;
      desc.ref.sampleCount = samples; 
      desc.ref.viewFormatCount = 0;
      desc.ref.viewFormats = nullptr;

      final texHandle =
          wgpu.wgpuDeviceCreateTexture(WebgpuRend.instance.device, desc);

      final viewDesc = arena<WGPUTextureViewDescriptor>();
      viewDesc.ref.label.data = nullptr;
      viewDesc.ref.label.length = 0;
      viewDesc.ref.format = WGPUTextureFormat.WGPUTextureFormat_Depth24Plus;
      viewDesc.ref.dimension =
          WGPUTextureViewDimension.WGPUTextureViewDimension_2D;
      viewDesc.ref.baseMipLevel = 0;
      viewDesc.ref.mipLevelCount = 1;
      viewDesc.ref.baseArrayLayer = 0;
      viewDesc.ref.arrayLayerCount = 1;
      viewDesc.ref.aspect = WGPUTextureAspect.WGPUTextureAspect_DepthOnly;

      final viewHandle = wgpu.wgpuTextureCreateView(texHandle, viewDesc);

      return GpuTexture._(
          texHandle.cast(), -1, texHandle, viewHandle, width, height, false);
    });
  }

  static GpuTexture createMsaa({required int width, required int height, int samples = 4}) {
    final wgpu = WebgpuRend.instance.wgpu;
    return using((arena) {
      final desc = arena<WGPUTextureDescriptor>();
      desc.ref.label.data = nullptr;
      desc.ref.size.width = width;
      desc.ref.size.height = height;
      desc.ref.size.depthOrArrayLayers = 1;
      desc.ref.mipLevelCount = 1;
      desc.ref.sampleCount = samples;
      desc.ref.dimension = WGPUTextureDimension.WGPUTextureDimension_2D;
      // MSAA textures are RenderAttachments only, never copied directly
      desc.ref.usage = WGPUTextureUsage_RenderAttachment; 
      desc.ref.format = kPreferredTextureFormat; // Must match screen format

      final handle = wgpu.wgpuDeviceCreateTexture(WebgpuRend.instance.device, desc);
      final viewHandle = wgpu.wgpuTextureCreateView(handle, nullptr);

      return GpuTexture._(handle.cast(), -1, handle, viewHandle, width, height, false);
    });
  }

  void beginAccess() {
    if (_disposed || !_isShared) return;
    WebgpuRend.instance.beginAccessInternal(_handle);
  }

  void endAccess() {
    if (_disposed || !_isShared) return;
    WebgpuRend.instance.endAccessInternal(_handle);
  }

  void present() {
    if (_disposed || !_isShared) return;
    WebgpuRend.instance.presentInternal(_handle);
  }

  void uploadRect(Uint8List data, Rect rect) {
    if (_disposed) return;
    final wgpu = WebgpuRend.instance.wgpu;
    final int x = rect.left.toInt();
    final int y = rect.top.toInt();
    final int w = rect.width.toInt();
    final int h = rect.height.toInt();

    // Basic validation
    if (data.length != w * h * 4) {
      throw ArgumentError(
          "Data size (${data.length}) does not match rect size ($w x $h x 4 = ${w * h * 4})");
    }

    beginAccess();

    using((arena) {
      final destination = arena<WGPUTexelCopyTextureInfo>();
      destination.ref.texture = texture;
      destination.ref.mipLevel = 0;
      destination.ref.origin.x = x;
      destination.ref.origin.y = y;
      destination.ref.origin.z = 0;
      destination.ref.aspect = WGPUTextureAspect.WGPUTextureAspect_All;

      final layout = arena<WGPUTexelCopyBufferLayout>();
      layout.ref.offset = 0;
      layout.ref.bytesPerRow = w * 4;
      layout.ref.rowsPerImage = h;

      final writeSize = arena<WGPUExtent3D>();
      writeSize.ref.width = w;
      writeSize.ref.height = h;
      writeSize.ref.depthOrArrayLayers = 1;

      final ptr = arena<Uint8>(data.length);
      ptr.asTypedList(data.length).setAll(0, data);

      wgpu.wgpuQueueWriteTexture(WebgpuRend.instance.queue, destination,
          ptr.cast(), data.length, layout, writeSize);
    });

    endAccess();
  }

  Future<Uint8List> download() async {
    if (_disposed) return Uint8List(0);
    final wgpu = WebgpuRend.instance.wgpu;
    wgpu.wgpuQueueSubmit(WebgpuRend.instance.queue, 0, nullptr);

    final size = width * height * 4;
    final int bytesPerRow = (width * 4 + 255) & ~255;
    final int paddedSize = bytesPerRow * height;

    final buffer = GpuBuffer.create(
        size: paddedSize,
        usage: WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

    final encoder = CommandEncoder();
    using((arena) {
      final src = arena<WGPUTexelCopyTextureInfo>();
      src.ref.texture = texture;
      src.ref.mipLevel = 0;
      src.ref.origin.x = 0;
      src.ref.origin.y = 0;
      src.ref.origin.z = 0;
      src.ref.aspect = WGPUTextureAspect.WGPUTextureAspect_All;

      final dst = arena<WGPUTexelCopyBufferInfo>();
      dst.ref.buffer = buffer.handle.cast();
      dst.ref.layout.offset = 0;
      dst.ref.layout.bytesPerRow = bytesPerRow;
      dst.ref.layout.rowsPerImage = height;

      final copySize = arena<WGPUExtent3D>();
      copySize.ref.width = width;
      copySize.ref.height = height;
      copySize.ref.depthOrArrayLayers = 1;

      wgpu.wgpuCommandEncoderCopyTextureToBuffer(
          encoder._handle, src, dst, copySize);
    });

    beginAccess();
    encoder.submit();
    endAccess();

    final paddedData = await buffer.mapRead();
    final compactData = Uint8List(size);

    for (int y = 0; y < height; y++) {
      final srcOffset = y * bytesPerRow;
      final dstOffset = y * (width * 4);
      for (int i = 0; i < (width * 4); i++) {
        compactData[dstOffset + i] = paddedData[srcOffset + i];
      }
    }

    buffer.dispose();
    return compactData;
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    final wgpu = WebgpuRend.instance.wgpu;
    wgpu.wgpuTextureViewRelease(view);
    if (_isShared) {
      _textureFinalizer.detach(this);
      WebgpuRend.instance.disposeTextureInternal(_handle);
    } else {
      wgpu.wgpuTextureRelease(texture);
    }
  }
}

class GpuBuffer extends GpuResource {
  final int size;
  final int usage;
  GpuBuffer._(super.handle, this.size, this.usage);

  static GpuBuffer create({required int size, required int usage}) {
    final wgpu = WebgpuRend.instance.wgpu;
    return using((arena) {
      final desc = arena<WGPUBufferDescriptor>();
      desc.ref.nextInChain = nullptr;
      desc.ref.label.data = nullptr;
      desc.ref.label.length = 0;
      desc.ref.size = size;
      desc.ref.usage = usage;
      desc.ref.mappedAtCreation = 0;
      final handle =
          wgpu.wgpuDeviceCreateBuffer(WebgpuRend.instance.device, desc);
      return GpuBuffer._(handle.cast(), size, usage);
    });
  }

  void update(Uint8List data) {
    using((arena) {
      final ptr = arena<Uint8>(data.length);
      ptr.asTypedList(data.length).setAll(0, data);
      updateRaw(ptr.cast(), data.length);
    });
  }

  void updateRaw(Pointer<Void> data, int dataSize) {
    WebgpuRend.instance.wgpu.wgpuQueueWriteBuffer(
        WebgpuRend.instance.queue, handle.cast(), 0, data, dataSize);
  }

  void updateRawOffset(Uint8List data, int bufferOffset) {
    using((arena) {
      final ptr = arena<Uint8>(data.length);
      ptr.asTypedList(data.length).setAll(0, data);
      
      WebgpuRend.instance.wgpu.wgpuQueueWriteBuffer(
          WebgpuRend.instance.queue, 
          handle.cast(), 
          bufferOffset, // Where in the GPU buffer to start writing
          ptr.cast(), 
          data.length   // How much to write
      );
    });
  }

    void uploadMatrix(Matrix4 matrix, {int offset = 0}) {
    final storage = matrix.storage;
    
    final f32 = Float32List(16);
    for (int i = 0; i < 16; i++) {
      f32[i] = storage[i];
    }
    
    final bytes = f32.buffer.asUint8List();
    updateRawOffset(bytes, offset);
  }

  Future<Uint8List> mapRead() async {
    if ((usage & WGPUBufferUsage_MapRead) == 0) return _readViaStaging();
    return _mapAsync(handle.cast());
  }

  Future<Uint8List> _readViaStaging() async {
    final wgpu = WebgpuRend.instance.wgpu;
    final staging = GpuBuffer.create(
        size: size, usage: WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    final encoder = wgpu.wgpuDeviceCreateCommandEncoder(
        WebgpuRend.instance.device, nullptr);
    wgpu.wgpuCommandEncoderCopyBufferToBuffer(
        encoder, handle.cast(), 0, staging.handle.cast(), 0, size);
    final cmd = wgpu.wgpuCommandEncoderFinish(encoder, nullptr);
    using((arena) {
      final ptr = arena<Pointer<Void>>();
      ptr.value = cmd.cast();
      wgpu.wgpuQueueSubmit(WebgpuRend.instance.queue, 1, ptr.cast());
    });
    final result = await staging._mapAsync(staging.handle.cast());
    staging.dispose();
    return result;
  }

  Future<Uint8List> _mapAsync(WGPUBuffer bufferHandle) async {
    final wgpu = WebgpuRend.instance.wgpu;
    final completer = Completer<void>();
    final callback = NativeCallable<WGPUBufferMapCallbackFunction>.listener(
        (int status, WGPUStringView msg, Pointer<Void> u1, Pointer<Void> u2) {
      if (status == WGPUMapAsyncStatus.WGPUMapAsyncStatus_Success.value) {
        completer.complete();
      } else {
        completer.completeError("Map Async Failed: $status");
      }
    });
    using((arena) {
      final callbackInfo = arena<WGPUBufferMapCallbackInfo>();
      callbackInfo.ref.mode =
          WGPUCallbackMode.WGPUCallbackMode_AllowSpontaneous;
      callbackInfo.ref.callback = callback.nativeFunction;
      callbackInfo.ref.userdata1 = nullptr;
      wgpu.wgpuBufferMapAsync(
          bufferHandle, WGPUMapMode_Read, 0, size, callbackInfo.ref);
    });
    while (!completer.isCompleted) {
      wgpu.wgpuDeviceTick(WebgpuRend.instance.device);
      await Future.delayed(Duration.zero);
    }
    callback.close();
    final ptr = wgpu.wgpuBufferGetConstMappedRange(bufferHandle, 0, size);
    final result = Uint8List.fromList(ptr.cast<Uint8>().asTypedList(size));
    wgpu.wgpuBufferUnmap(bufferHandle);
    return result;
  }

  void dispose() {
    WebgpuRend.instance.wgpu.wgpuBufferRelease(handle.cast());
  }
}

class GpuSampler extends GpuResource {
  GpuSampler._(super.handle);

  static GpuSampler create({
    WGPUAddressMode addressMode = WGPUAddressMode.WGPUAddressMode_ClampToEdge,
    WGPUFilterMode filter = WGPUFilterMode.WGPUFilterMode_Linear,
    WGPUCompareFunction compare =
        WGPUCompareFunction.WGPUCompareFunction_Undefined,
  }) {
    final wgpu = WebgpuRend.instance.wgpu;
    return using((arena) {
      final desc = arena<WGPUSamplerDescriptor>();
      desc.ref.label.data = nullptr;
      desc.ref.label.length = 0;
      desc.ref.magFilter = filter;
      desc.ref.minFilter = filter;
      desc.ref.mipmapFilter = filter == WGPUFilterMode.WGPUFilterMode_Linear
          ? WGPUMipmapFilterMode.WGPUMipmapFilterMode_Linear
          : WGPUMipmapFilterMode.WGPUMipmapFilterMode_Nearest;
      desc.ref.addressModeU = addressMode;
      desc.ref.addressModeV = addressMode;
      desc.ref.addressModeW = addressMode;
      desc.ref.lodMinClamp = 0.0;
      desc.ref.lodMaxClamp = 1000.0;
      desc.ref.compare = compare;
      desc.ref.maxAnisotropy = 1;
      final handle =
          wgpu.wgpuDeviceCreateSampler(WebgpuRend.instance.device, desc);
      return GpuSampler._(handle.cast());
    });
  }

  void dispose() => WebgpuRend.instance.wgpu.wgpuSamplerRelease(handle.cast());
}

class GpuShader extends GpuResource {
  GpuShader._(super.handle);
  static GpuShader create(String source) {
    final wgpu = WebgpuRend.instance.wgpu;
    return using((arena) {
      final wgslDesc = arena<WGPUShaderSourceWGSL>();
      wgslDesc.ref.chain.sType = WGPUSType.WGPUSType_ShaderSourceWGSL;
      wgslDesc.ref.chain.next = nullptr;
      wgslDesc.ref.code = _createStringView(arena, source);
      final desc = arena<WGPUShaderModuleDescriptor>();
      desc.ref.nextInChain = wgslDesc.cast();
      desc.ref.label.data = nullptr;
      desc.ref.label.length = 0;
      final handle =
          wgpu.wgpuDeviceCreateShaderModule(WebgpuRend.instance.device, desc);
      return GpuShader._(handle.cast());
    });
  }

  void dispose() =>
      WebgpuRend.instance.wgpu.wgpuShaderModuleRelease(handle.cast());
}

class GpuRenderPipeline extends GpuResource {
  GpuRenderPipeline._(super.handle);

  static GpuRenderPipeline create({
    required GpuShader vertexShader,
    required GpuShader fragmentShader,
    required List<VertexBufferLayout> bufferLayouts,
    BlendMode blendMode = BlendMode.opaque, 
    bool enableDepth = false,
    WGPUTextureFormat depthFormat =
        WGPUTextureFormat.WGPUTextureFormat_Depth24Plus,
    String vertexEntryPoint = "main",
    String fragmentEntryPoint = "main",
    int sampleCount = 1,
    WGPUTextureFormat? targetFormat,
    WGPUPrimitiveTopology topology =
        WGPUPrimitiveTopology.WGPUPrimitiveTopology_TriangleStrip,
    WGPUCullMode cullMode = WGPUCullMode.WGPUCullMode_None,
    WGPUFrontFace frontFace = WGPUFrontFace.WGPUFrontFace_CCW,
  }) {
    final wgpu = WebgpuRend.instance.wgpu;
    final format = targetFormat ?? kPreferredTextureFormat;

    return using((arena) {
      // Vertex State Setup
      final vertexState = arena<WGPUVertexState>();
      vertexState.ref.module = vertexShader.handle.cast();
      vertexState.ref.entryPoint = _createStringView(arena, vertexEntryPoint);
      vertexState.ref.constantCount = 0;

      if (bufferLayouts.isNotEmpty) {
        final layouts = arena<WGPUVertexBufferLayout>(bufferLayouts.length);
        int totalAttrs = 0;
        for (var l in bufferLayouts) {
          totalAttrs += l.attributes.length;
        }
        final attrs = arena<WGPUVertexAttribute>(totalAttrs);

        int attrIdx = 0;

        for (int i = 0; i < bufferLayouts.length; i++) {
          final def = bufferLayouts[i];
          final layout = layouts.elementAt(i);
          layout.ref.arrayStride = def.arrayStride;
          layout.ref.stepMode = def.stepMode;
          layout.ref.attributeCount = def.attributes.length;
          layout.ref.attributes = attrs.elementAt(attrIdx);

          for (int j = 0; j < def.attributes.length; j++) {
            final dartAttr = def.attributes[j];
            final nativeAttr = attrs.elementAt(attrIdx + j);
            nativeAttr.ref.format = dartAttr.format;
            nativeAttr.ref.offset = dartAttr.offset;
            nativeAttr.ref.shaderLocation = dartAttr.shaderLocation;
          }
          attrIdx += def.attributes.length;
        }
        vertexState.ref.bufferCount = bufferLayouts.length;
        vertexState.ref.buffers = layouts;
      } else {
        vertexState.ref.bufferCount = 0;
        vertexState.ref.buffers = nullptr;
      }

      // Fragment State Setup
      final fragmentState = arena<WGPUFragmentState>();
      fragmentState.ref.module = fragmentShader.handle.cast();
      fragmentState.ref.entryPoint = _createStringView(arena, fragmentEntryPoint);
      fragmentState.ref.constantCount = 0;
      fragmentState.ref.targetCount = 1;

      final target = arena<WGPUColorTargetState>();
      target.ref.format = format;
      target.ref.writeMask = WGPUColorWriteMask_All;

      if (blendMode == BlendMode.opaque) {
        target.ref.blend = nullptr;
      } else {
        final blend = arena<WGPUBlendState>();
        
        // Default assignments
        var srcFactorColor = WGPUBlendFactor.WGPUBlendFactor_One;
        var dstFactorColor = WGPUBlendFactor.WGPUBlendFactor_Zero;
        var opColor = WGPUBlendOperation.WGPUBlendOperation_Add;
        
        var srcFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_One;
        var dstFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_Zero;
        var opAlpha = WGPUBlendOperation.WGPUBlendOperation_Add;

        switch (blendMode) {
          case BlendMode.alpha:
            // Final = (Src * SrcAlpha) + (Dst * (1 - SrcAlpha))
            srcFactorColor = WGPUBlendFactor.WGPUBlendFactor_SrcAlpha;
            dstFactorColor = WGPUBlendFactor.WGPUBlendFactor_OneMinusSrcAlpha;
            srcFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_One;
            dstFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_OneMinusSrcAlpha;
            break;

          case BlendMode.add:
            // Final = Src + Dst
            srcFactorColor = WGPUBlendFactor.WGPUBlendFactor_One;
            dstFactorColor = WGPUBlendFactor.WGPUBlendFactor_One;
            srcFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_One;
            dstFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_One;
            break;

          case BlendMode.max:
            // Final = Max(Src, Dst) - Factors are ignored in Max/Min
            opColor = WGPUBlendOperation.WGPUBlendOperation_Max;
            opAlpha = WGPUBlendOperation.WGPUBlendOperation_Max;
            break;

          case BlendMode.min:
            // Final = Min(Src, Dst)
            opColor = WGPUBlendOperation.WGPUBlendOperation_Min;
            opAlpha = WGPUBlendOperation.WGPUBlendOperation_Min;
            break;

          case BlendMode.erase:
            // Final = Dst * (1 - SrcAlpha)
            srcFactorColor = WGPUBlendFactor.WGPUBlendFactor_Zero;
            dstFactorColor = WGPUBlendFactor.WGPUBlendFactor_OneMinusSrcAlpha;
            srcFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_Zero;
            dstFactorAlpha = WGPUBlendFactor.WGPUBlendFactor_OneMinusSrcAlpha;
            break;
            
          case BlendMode.opaque:
            break; // Handled above
        }

        blend.ref.color.srcFactor = srcFactorColor;
        blend.ref.color.dstFactor = dstFactorColor;
        blend.ref.color.operation = opColor;
        blend.ref.alpha.srcFactor = srcFactorAlpha;
        blend.ref.alpha.dstFactor = dstFactorAlpha;
        blend.ref.alpha.operation = opAlpha;
        
        target.ref.blend = blend;
      }

      fragmentState.ref.targets = target;

      // Pipeline Descriptor
      final desc = arena<WGPURenderPipelineDescriptor>();
      desc.ref.label.data = nullptr;
      desc.ref.label.length = 0;
      desc.ref.layout = nullptr;
      desc.ref.vertex = vertexState.ref;
      desc.ref.fragment = fragmentState;

      desc.ref.primitive.topology = topology;
      desc.ref.primitive.stripIndexFormat = WGPUIndexFormat.WGPUIndexFormat_Undefined;
      desc.ref.primitive.frontFace = frontFace;
      desc.ref.primitive.cullMode = cullMode;

      if (enableDepth) {
        final ds = arena<WGPUDepthStencilState>();
        ds.ref.format = depthFormat;
        ds.ref.depthWriteEnabled = WGPUOptionalBool.WGPUOptionalBool_True;
        ds.ref.depthCompare = WGPUCompareFunction.WGPUCompareFunction_Less;
        ds.ref.stencilReadMask = 0;
        ds.ref.stencilWriteMask = 0;
        desc.ref.depthStencil = ds;
      } else {
        desc.ref.depthStencil = nullptr;
      }

      desc.ref.multisample.count = sampleCount;
      desc.ref.multisample.mask = 0xFFFFFFFF;
      desc.ref.multisample.alphaToCoverageEnabled = 0;

      final handle = wgpu.wgpuDeviceCreateRenderPipeline(WebgpuRend.instance.device, desc);
      return GpuRenderPipeline._(handle.cast());
    });
  }

  WGPUBindGroup createBindGroup(int index, List<Object> resources) {
    final layout = WebgpuRend.instance.wgpu
        .wgpuRenderPipelineGetBindGroupLayout(handle.cast(), index);
    final group = _createBindGroupHelper(layout, resources);
    WebgpuRend.instance.wgpu.wgpuBindGroupLayoutRelease(layout);
    return group;
  }

  void dispose() => WebgpuRend.instance.wgpu.wgpuRenderPipelineRelease(handle.cast());
}

class GpuComputePipeline extends GpuResource {
  GpuComputePipeline._(super.handle);
  static GpuComputePipeline create(GpuShader shader,
      {String entryPoint = "main"}) {
    final wgpu = WebgpuRend.instance.wgpu;
    return using((arena) {
      final desc = arena<WGPUComputePipelineDescriptor>();
      desc.ref.label.data = nullptr;
      desc.ref.label.length = 0;
      desc.ref.layout = nullptr;
      desc.ref.compute.module = shader.handle.cast();
      desc.ref.compute.entryPoint = _createStringView(arena, entryPoint);
      desc.ref.compute.constantCount = 0;
      final handle = wgpu.wgpuDeviceCreateComputePipeline(
          WebgpuRend.instance.device, desc);
      return GpuComputePipeline._(handle.cast());
    });
  }

  WGPUBindGroup createBindGroup(int index, List<Object> resources) {
    final layout = WebgpuRend.instance.wgpu
        .wgpuComputePipelineGetBindGroupLayout(handle.cast(), index);
    final group = _createBindGroupHelper(layout, resources);
    WebgpuRend.instance.wgpu.wgpuBindGroupLayoutRelease(layout);
    return group;
  }

  void dispose() =>
      WebgpuRend.instance.wgpu.wgpuComputePipelineRelease(handle.cast());
}

class CommandEncoder {
  final WGPUCommandEncoder _handle;
  final WebGpuBindings _wgpu = WebgpuRend.instance.wgpu;
  CommandEncoder()
      : _handle = WebgpuRend.instance.wgpu.wgpuDeviceCreateCommandEncoder(
            WebgpuRend.instance.device, nullptr);

  void copyTextureToTexture(
      {required GpuTexture from, required GpuTexture to}) {
    using((arena) {
      final srcInfo = arena<WGPUTexelCopyTextureInfo>();
      srcInfo.ref.texture = from.texture;
      srcInfo.ref.mipLevel = 0;
      srcInfo.ref.origin = (arena<WGPUOrigin3D>()
            ..ref.x = 0
            ..ref.y = 0
            ..ref.z = 0)
          .ref;
      srcInfo.ref.aspect = WGPUTextureAspect.WGPUTextureAspect_All;
      final dstInfo = arena<WGPUTexelCopyTextureInfo>();
      dstInfo.ref.texture = to.texture;
      dstInfo.ref.mipLevel = 0;
      dstInfo.ref.origin = (arena<WGPUOrigin3D>()
            ..ref.x = 0
            ..ref.y = 0
            ..ref.z = 0)
          .ref;
      dstInfo.ref.aspect = WGPUTextureAspect.WGPUTextureAspect_All;
      final extent = arena<WGPUExtent3D>();
      extent.ref.width = from.width;
      extent.ref.height = from.height;
      extent.ref.depthOrArrayLayers = 1;
      _wgpu.wgpuCommandEncoderCopyTextureToTexture(
          _handle, srcInfo, dstInfo, extent);
    });
  }

  RenderPassEncoder beginRenderPass(
    GpuTexture texture, {
    Color? clearColor,
    int sampleCount = 1,
    GpuTexture? msaaTexture,
    GpuTexture? depthTexture,
    WGPULoadOp loadOp = WGPULoadOp.WGPULoadOp_Load,
    WGPUStoreOp storeOp = WGPUStoreOp.WGPUStoreOp_Store,
  }) {
    final scratch = _Scratchpad.instance;
    final colorAttr = scratch.colorAttachment;

     if (sampleCount > 1) {
      if (msaaTexture == null) {
        throw ArgumentError("If sampleCount > 1, msaaTexture must be provided");
      }
      colorAttr.ref.view = msaaTexture.view;
      colorAttr.ref.resolveTarget = texture.view;
      colorAttr.ref.storeOp = WGPUStoreOp.WGPUStoreOp_Discard;
    } else {
      // Standard 1x rendering
      colorAttr.ref.view = texture.view;
      colorAttr.ref.resolveTarget = nullptr;
      colorAttr.ref.storeOp = WGPUStoreOp.WGPUStoreOp_Store;
    }

    colorAttr.ref.loadOp =
        clearColor != null ? WGPULoadOp.WGPULoadOp_Clear : loadOp;
    colorAttr.ref.depthSlice = 0xFFFFFFFF;
    if (clearColor != null) {
      colorAttr.ref.clearValue.r = clearColor.red / 255.0;
      colorAttr.ref.clearValue.g = clearColor.green / 255.0;
      colorAttr.ref.clearValue.b = clearColor.blue / 255.0;
      colorAttr.ref.clearValue.a = clearColor.opacity;
    }

    final desc = scratch.renderPassDesc;
    desc.ref.label.data = nullptr;
    desc.ref.label.length = 0;
    desc.ref.colorAttachmentCount = 1;
    desc.ref.colorAttachments = colorAttr;

    if (depthTexture != null) {
      final depthAttr = scratch.depthStencilAttachment;
      depthAttr.ref.view = depthTexture.view;
      depthAttr.ref.depthClearValue = 1.0;
      depthAttr.ref.depthLoadOp = WGPULoadOp.WGPULoadOp_Clear;
      depthAttr.ref.depthStoreOp = WGPUStoreOp.WGPUStoreOp_Store;
      depthAttr.ref.stencilLoadOp = WGPULoadOp.WGPULoadOp_Undefined;
      depthAttr.ref.stencilStoreOp = WGPUStoreOp.WGPUStoreOp_Undefined;
      desc.ref.depthStencilAttachment = depthAttr;
    } else {
      desc.ref.depthStencilAttachment = nullptr;
    }

    desc.ref.timestampWrites = nullptr;
    desc.ref.occlusionQuerySet = nullptr;
    final passHandle = _wgpu.wgpuCommandEncoderBeginRenderPass(_handle, desc);
    return RenderPassEncoder(passHandle);
  }

  ComputePassEncoder beginComputePass() {
    final passHandle =
        _wgpu.wgpuCommandEncoderBeginComputePass(_handle, nullptr);
    return ComputePassEncoder(passHandle);
  }

  void submit() {
    final cmdBuf = _wgpu.wgpuCommandEncoderFinish(_handle, nullptr);
    using((arena) {
      final ptr = arena<Pointer<Void>>();
      ptr.value = cmdBuf.cast();
      _wgpu.wgpuQueueSubmit(WebgpuRend.instance.queue, 1, ptr.cast());
    });

    _wgpu.wgpuCommandBufferRelease(cmdBuf);
    _wgpu.wgpuCommandEncoderRelease(_handle);

  }
}

class RenderPassEncoder {
  final WGPURenderPassEncoder _handle;
  final WebGpuBindings _wgpu = WebgpuRend.instance.wgpu;
  RenderPassEncoder(this._handle);
  void bindPipeline(GpuRenderPipeline pipeline) {
    _wgpu.wgpuRenderPassEncoderSetPipeline(_handle, pipeline.handle.cast());
  }

  void setBindGroup(int index, WGPUBindGroup group) {
    _wgpu.wgpuRenderPassEncoderSetBindGroup(_handle, index, group, 0, nullptr);
  }

  void setVertexBuffer(int slot, GpuBuffer buffer) {
    _wgpu.wgpuRenderPassEncoderSetVertexBuffer(
        _handle, slot, buffer.handle.cast(), 0, buffer.size);
  }

  void draw(int vertexCount) =>
      _wgpu.wgpuRenderPassEncoderDraw(_handle, vertexCount, 1, 0, 0);
  void drawInstanced(int vertexCount, int instanceCount) => _wgpu
      .wgpuRenderPassEncoderDraw(_handle, vertexCount, instanceCount, 0, 0);

  void setIndexBuffer(GpuBuffer buffer, WGPUIndexFormat format, [int offset = 0, int size = 0]) {
    // If size is 0, use whole buffer
    final effectiveSize = size == 0 ? buffer.size - offset : size;
    _wgpu.wgpuRenderPassEncoderSetIndexBuffer(_handle, buffer.handle.cast(), format, offset, effectiveSize);
  }

  void drawIndexed(int indexCount, [int instanceCount = 1, int firstIndex = 0, int baseVertex = 0, int firstInstance = 0]) {
    _wgpu.wgpuRenderPassEncoderDrawIndexed(_handle, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
  }
  void end() => _wgpu.wgpuRenderPassEncoderEnd(_handle);
}

class ComputePassEncoder {
  final WGPUComputePassEncoder _handle;
  final WebGpuBindings _wgpu = WebgpuRend.instance.wgpu;
  ComputePassEncoder(this._handle);
  void bindPipeline(GpuComputePipeline pipeline) {
    _wgpu.wgpuComputePassEncoderSetPipeline(_handle, pipeline.handle.cast());
  }

  void setBindGroup(int index, WGPUBindGroup group) {
    _wgpu.wgpuComputePassEncoderSetBindGroup(_handle, index, group, 0, nullptr);
  }

  void dispatch(int x, [int y = 1, int z = 1]) =>
      _wgpu.wgpuComputePassEncoderDispatchWorkgroups(_handle, x, y, z);
  void end() => _wgpu.wgpuComputePassEncoderEnd(_handle);
}

enum BlendMode {
  /// No blending. Replaces destination pixels.
  opaque,

  /// alpha blending: (Src * SrcAlpha) + (Dst * (1 - SrcAlpha))
  alpha,

  /// Additive blending: Src + Dst
  add,

  /// Retains the maximum component value: Max(Src, Dst)
  max,

  /// Retains the minimum component value: Min(Src, Dst)
  min,

  /// Erases destination based on source alpha: Dst * (1 - SrcAlpha)
  erase,
}
