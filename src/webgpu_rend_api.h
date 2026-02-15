#ifndef WEBGPU_REND_API_H
#define WEBGPU_REND_API_H

#include <stdint.h>

#if _WIN32
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT
#endif

// Opaque handle for our C++ texture wrapper
typedef void* WebgpuRendTexture;

#ifdef __cplusplus
extern "C" {
#endif

// Initialization
// Returns the WGPUDevice pointer
API_EXPORT void* webgpu_rend_init(void* texture_registrar);

// Helper to look up WebGPU functions
API_EXPORT void* webgpu_rend_get_proc_address(const char* procName);

// Texture Bridge
// Creates a Flutter Texture backed by a WebGPU Texture
API_EXPORT WebgpuRendTexture webgpu_rend_create_texture(int32_t width, int32_t height);
API_EXPORT int64_t webgpu_rend_get_texture_id(WebgpuRendTexture handle);
// Returns the WGPUTexture pointer
API_EXPORT void* webgpu_rend_get_wgpu_texture(WebgpuRendTexture handle);
// Returns a WGPUTextureView pointer (default view)
API_EXPORT void* webgpu_rend_get_wgpu_texture_view(WebgpuRendTexture handle);
API_EXPORT void webgpu_rend_present_texture(WebgpuRendTexture handle);
API_EXPORT void webgpu_rend_dispose_texture(WebgpuRendTexture handle);

#ifdef __cplusplus
}
#endif

#endif  // WEBGPU_REND_API_H