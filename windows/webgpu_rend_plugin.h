#ifndef FLUTTER_PLUGIN_WEBGPU_REND_PLUGIN_H_
#define FLUTTER_PLUGIN_WEBGPU_REND_PLUGIN_H_

#include <d3d11.h>
#include <dawn/webgpu_cpp.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/texture_registrar.h>
#include <wrl/client.h>

#include <memory>

#include "../src/webgpu_rend_api.h"

namespace webgpu_rend {

struct GpuTextureObject {
    GpuTextureObject(int width, int height, flutter::TextureRegistrar& registrar, Microsoft::WRL::ComPtr<ID3D11Device> device, wgpu::Device wgpu_device);
    ~GpuTextureObject();

    int width, height;
    int64_t texture_id;
    flutter::TextureRegistrar& texture_registrar;

    Microsoft::WRL::ComPtr<ID3D11Device> d3d_device;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d_texture;
    std::unique_ptr<flutter::TextureVariant> texture_variant;
    std::unique_ptr<FlutterDesktopGpuSurfaceDescriptor> surface_descriptor;

    std::unique_ptr<wgpu::SharedTextureMemory> shared_memory;
    wgpu::Texture webgpu_texture;
    wgpu::TextureView default_view;
};

class WebgpuRendPlugin {
   public:
    static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);
};

}  // namespace webgpu_rend

#endif