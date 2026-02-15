#include "webgpu_rend_ffi.h"

#include <flutter/plugin_registrar_windows.h>

#include <map>
#include <memory>

#include "webgpu_rend_plugin.h"

// Global state for our FFI plugin
static flutter::PluginRegistrarWindows* g_registrar = nullptr;
static std::map<int64_t, std::unique_ptr<webgpu_rend::PixelTextureObject>> g_textures;

void webgpu_rend_init_ffi(void* registrar) {
    g_registrar = static_cast<flutter::PluginRegistrarWindows*>(registrar);
}

int64_t webgpu_rend_create_texture(int32_t width, int32_t height) {
    if (!g_registrar || !g_registrar->texture_registrar()) {
        return -1;  // Not initialized
    }

    auto texture_object = std::make_unique<webgpu_rend::PixelTextureObject>(
        width, height, *g_registrar->texture_registrar());

    int64_t texture_id = texture_object->get_texture_id();
    g_textures[texture_id] = std::move(texture_object);

    return texture_id;
}

uint8_t* webgpu_rend_get_pixel_buffer(int64_t texture_id) {
    auto it = g_textures.find(texture_id);
    if (it != g_textures.end()) {
        return it->second->get_pixels().data();
    }
    return nullptr;
}

void webgpu_rend_invalidate_texture(int64_t texture_id) {
    if (!g_registrar) return;

    auto it = g_textures.find(texture_id);
    if (it != g_textures.end()) {
        it->second->invalidate();
    }
}

void webgpu_rend_dispose_texture(int64_t texture_id) {
    auto it = g_textures.find(texture_id);
    if (it != g_textures.end()) {
        g_textures.erase(it);
    }
}