#include "webgpu_rend_plugin.h"

#include <dawn/dawn_proc_table.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu.h>

#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

using namespace webgpu_rend;
using Microsoft::WRL::ComPtr;

// Globals
static ComPtr<ID3D11Device> g_d3d_device;
static flutter::TextureRegistrar* g_texture_registrar = nullptr;
static std::unique_ptr<dawn::native::Instance> g_dawn_instance;
static wgpu::Device g_wgpu_device;
static wgpu::Queue g_wgpu_queue;
static std::map<WebgpuRendTexture, std::unique_ptr<GpuTextureObject>> g_textures;
static std::mutex g_mutex;

// D3D11 Helper
bool InitializeD3D11() {
    if (g_d3d_device) return true;
    UINT creation_flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL feature_levels[] = {D3D_FEATURE_LEVEL_11_0};
    ComPtr<ID3D11DeviceContext> ctx;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, creation_flags, feature_levels, 1, D3D11_SDK_VERSION, &g_d3d_device, nullptr, &ctx);
    return SUCCEEDED(hr);
}

// Dawn Error Callback
void PrintDeviceError(WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2) {
    std::cerr << "Dawn Error (" << type << "): " << std::string(message.data, message.length) << std::endl;
}

void InitializeDawn() {
    if (g_wgpu_device) return;

    g_dawn_instance = std::make_unique<dawn::native::Instance>();

    WGPURequestAdapterOptions options = {};
    options.powerPreference = WGPUPowerPreference_HighPerformance;

    std::vector<dawn::native::Adapter> adapters = g_dawn_instance->EnumerateAdapters(&options);
    if (adapters.empty()) {
        std::cerr << "No WebGPU adapters found." << std::endl;
        return;
    }

    dawn::native::Adapter chosenAdapter = adapters[0];
    for (const auto& adapter : adapters) {
        WGPUAdapterInfo info = {};
        wgpuAdapterGetInfo(adapter.Get(), &info);
        if (info.adapterType == WGPUAdapterType_DiscreteGPU) {
            chosenAdapter = adapter;
            break;
        }
    }

    WGPUDeviceDescriptor deviceDesc = {};

    WGPUFeatureName requiredFeatures[] = {
        WGPUFeatureName_SharedTextureMemoryDXGISharedHandle};
    deviceDesc.requiredFeatures = requiredFeatures;
    deviceDesc.requiredFeatureCount = 1;

    WGPUUncapturedErrorCallbackInfo errorCallbackInfo = {};
    errorCallbackInfo.callback = PrintDeviceError;
    errorCallbackInfo.userdata1 = nullptr;
    errorCallbackInfo.userdata2 = nullptr;
    deviceDesc.uncapturedErrorCallbackInfo = errorCallbackInfo;

    WGPUDevice cDevice = chosenAdapter.CreateDevice(&deviceDesc);
    if (!cDevice) {
        std::cerr << "Failed to create WebGPU Device." << std::endl;
        return;
    }

    g_wgpu_device = wgpu::Device::Acquire(cDevice);
    g_wgpu_queue = g_wgpu_device.GetQueue();
}

GpuTextureObject::GpuTextureObject(int w, int h, flutter::TextureRegistrar& registrar, ComPtr<ID3D11Device> device, wgpu::Device wgpu_dev)
    : width(w), height(h), texture_registrar(registrar), d3d_device(device), texture_id(-1) {
    D3D11_TEXTURE2D_DESC d3d_desc = {};
    d3d_desc.Width = width;
    d3d_desc.Height = height;
    d3d_desc.MipLevels = 1;
    d3d_desc.ArraySize = 1;
    d3d_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    d3d_desc.SampleDesc.Count = 1;
    d3d_desc.Usage = D3D11_USAGE_DEFAULT;
    d3d_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    d3d_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    if (FAILED(d3d_device->CreateTexture2D(&d3d_desc, nullptr, &d3d_texture))) {
        throw std::runtime_error("Failed to create D3D11 texture");
    }

    ComPtr<IDXGIResource> dxgi_resource;
    d3d_texture.As(&dxgi_resource);
    HANDLE shared_handle;
    dxgi_resource->GetSharedHandle(&shared_handle);

    surface_descriptor = std::make_unique<FlutterDesktopGpuSurfaceDescriptor>();
    surface_descriptor->struct_size = sizeof(FlutterDesktopGpuSurfaceDescriptor);
    surface_descriptor->handle = shared_handle;
    surface_descriptor->width = width;
    surface_descriptor->height = height;
    surface_descriptor->visible_width = width;
    surface_descriptor->visible_height = height;
    surface_descriptor->format = kFlutterDesktopPixelFormatBGRA8888;

    texture_variant = std::make_unique<flutter::TextureVariant>(flutter::GpuSurfaceTexture(
        kFlutterDesktopGpuSurfaceTypeDxgiSharedHandle,
        [this](auto, auto) { return this->surface_descriptor.get(); }));

    texture_id = texture_registrar.RegisterTexture(texture_variant.get());

    wgpu::SharedTextureMemoryDXGISharedHandleDescriptor handle_desc{};
    handle_desc.handle = shared_handle;
    wgpu::SharedTextureMemoryDescriptor stm_desc{};
    stm_desc.nextInChain = &handle_desc;
    stm_desc.label = "FlutterImportedTexture";

    shared_memory = std::make_unique<wgpu::SharedTextureMemory>(wgpu_dev.ImportSharedTextureMemory(&stm_desc));

    wgpu::TextureDescriptor tex_desc{};
    tex_desc.dimension = wgpu::TextureDimension::e2D;
    tex_desc.size = {(uint32_t)width, (uint32_t)height, 1};
    tex_desc.format = wgpu::TextureFormat::BGRA8Unorm;

    tex_desc.usage = wgpu::TextureUsage::RenderAttachment |
                     wgpu::TextureUsage::CopyDst |
                     wgpu::TextureUsage::TextureBinding |
                     wgpu::TextureUsage::CopySrc;

    webgpu_texture = shared_memory->CreateTexture(&tex_desc);
    default_view = webgpu_texture.CreateView();
}

GpuTextureObject::~GpuTextureObject() {
    texture_registrar.UnregisterTexture(texture_id);
}

void WebgpuRendPlugin::RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar) {
    g_texture_registrar = registrar->texture_registrar();
    InitializeD3D11();
}

extern "C" {

API_EXPORT void* webgpu_rend_get_proc_address(const char* procName) {
    WGPUStringView view;
    view.data = procName;
    view.length = std::strlen(procName);
    return (void*)dawn::native::GetProcs().getProcAddress(view);
}

API_EXPORT void* webgpu_rend_init(void* registrar) {
    InitializeDawn();
    return g_wgpu_device.Get();
}

API_EXPORT WebgpuRendTexture webgpu_rend_create_texture(int32_t width, int32_t height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_wgpu_device) return nullptr;
    try {
        auto tex = std::make_unique<GpuTextureObject>(width, height, *g_texture_registrar, g_d3d_device, g_wgpu_device);
        WebgpuRendTexture handle = tex.get();
        g_textures[handle] = std::move(tex);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

API_EXPORT int64_t webgpu_rend_get_texture_id(WebgpuRendTexture t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_textures.count(t) ? g_textures.at(t)->texture_id : -1;
}

API_EXPORT void* webgpu_rend_get_wgpu_texture(WebgpuRendTexture t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_textures.count(t) ? g_textures.at(t)->webgpu_texture.Get() : nullptr;
}

API_EXPORT void* webgpu_rend_get_wgpu_texture_view(WebgpuRendTexture t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_textures.count(t) ? g_textures.at(t)->default_view.Get() : nullptr;
}

API_EXPORT void webgpu_rend_texture_begin_access(WebgpuRendTexture t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_textures.count(t)) return;
    auto& tex = g_textures.at(t);

    wgpu::SharedTextureMemoryBeginAccessDescriptor desc = {};
    desc.initialized = true;
    desc.fenceCount = 0;

    tex->shared_memory->BeginAccess(tex->webgpu_texture, &desc);
}

API_EXPORT void webgpu_rend_texture_end_access(WebgpuRendTexture t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_textures.count(t)) return;
    auto& tex = g_textures.at(t);

    wgpu::SharedTextureMemoryEndAccessState state = {};
    tex->shared_memory->EndAccess(tex->webgpu_texture, &state);
}

API_EXPORT void webgpu_rend_present_texture(WebgpuRendTexture t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_textures.count(t)) return;

    g_wgpu_queue.Submit(0, nullptr);
    g_textures.at(t)->texture_registrar.MarkTextureFrameAvailable(g_textures.at(t)->texture_id);
}

API_EXPORT void webgpu_rend_dispose_texture(WebgpuRendTexture t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_textures.erase(t);
}

}  // extern C