#include <android/log.h>
#include <android/native_window_jni.h>
#include <dawn/dawn_proc_table.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp.h>
#include <jni.h>

#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#define LOG_TAG "WebgpuRend"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// JNI Globals
static JavaVM* g_vm = nullptr;
static jclass g_plugin_class = nullptr;
static jmethodID g_create_texture_mid = nullptr;
static jmethodID g_get_id_mid = nullptr;
static jmethodID g_get_surface_mid = nullptr;
static jmethodID g_dispose_mid = nullptr;

// Dawn Globals
static std::unique_ptr<dawn::native::Instance> g_instance;
static wgpu::Device g_device;
static wgpu::Queue g_queue;
static std::mutex g_mutex;

JNIEnv* GetEnv() {
    JNIEnv* env;
    if (g_vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        g_vm->AttachCurrentThread(&env, nullptr);
    }
    return env;
}

void PrintDeviceError(WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2) {
    LOGE("Dawn Error (%d): %.*s", type, (int)message.length, message.data);
}

struct AndroidTextureObject {
    int width;
    int height;
    int handle;
    int64_t flutter_texture_id;

    ANativeWindow* window = nullptr;
    wgpu::Surface surface = nullptr;

    wgpu::Texture working_texture = nullptr;
    wgpu::TextureView working_view = nullptr;

    AndroidTextureObject(int w, int h) : width(w), height(h) {
        JNIEnv* env = GetEnv();
        handle = env->CallStaticIntMethod(g_plugin_class, g_create_texture_mid, w, h);
        flutter_texture_id = env->CallStaticLongMethod(g_plugin_class, g_get_id_mid, handle);

        jobject jSurface = env->CallStaticObjectMethod(g_plugin_class, g_get_surface_mid, handle);
        window = ANativeWindow_fromSurface(env, jSurface);
        env->DeleteLocalRef(jSurface);

        WGPUSurfaceSourceAndroidNativeWindow androidDesc = {};
        androidDesc.chain.sType = WGPUSType_SurfaceSourceAndroidNativeWindow;
        androidDesc.window = window;

        WGPUSurfaceDescriptor surfDesc = {};
        surfDesc.nextInChain = (WGPUChainedStruct*)&androidDesc;

        WGPUSurface cSurface = wgpuInstanceCreateSurface(g_instance->Get(), &surfDesc);
        surface = wgpu::Surface::Acquire(cSurface);

        wgpu::SurfaceConfiguration config = {};
        config.device = g_device;
        config.format = wgpu::TextureFormat::RGBA8Unorm;
        config.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopyDst;
        config.width = width;
        config.height = height;
        config.presentMode = wgpu::PresentMode::Fifo;
        surface.Configure(&config);

        wgpu::TextureDescriptor workDesc = {};
        workDesc.label = {nullptr, 0};
        workDesc.size = {(uint32_t)width, (uint32_t)height, 1};
        workDesc.dimension = wgpu::TextureDimension::e2D;
        workDesc.format = wgpu::TextureFormat::RGBA8Unorm;
        workDesc.usage = wgpu::TextureUsage::RenderAttachment |
                         wgpu::TextureUsage::TextureBinding |
                         wgpu::TextureUsage::StorageBinding |
                         wgpu::TextureUsage::CopySrc |
                         wgpu::TextureUsage::CopyDst;

        working_texture = g_device.CreateTexture(&workDesc);
        working_view = working_texture.CreateView();
    }

    ~AndroidTextureObject() {
        JNIEnv* env = GetEnv();
        env->CallStaticVoidMethod(g_plugin_class, g_dispose_mid, handle);
        if (window) ANativeWindow_release(window);
    }
};

static std::map<void*, std::unique_ptr<AndroidTextureObject>> g_textures;

// FFI Exports

extern "C" {

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif

API_EXPORT void* webgpu_rend_get_proc_address(const char* procName) {
    WGPUStringView view;
    view.data = procName;
    view.length = std::strlen(procName);

    return (void*)dawn::native::GetProcs().getProcAddress(view);
}

API_EXPORT void* webgpu_rend_init(void*) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_device) return g_device.Get();

    g_instance = std::make_unique<dawn::native::Instance>();

    WGPURequestAdapterOptions options = {};
    options.backendType = WGPUBackendType_Vulkan;

    std::vector<dawn::native::Adapter> adapters = g_instance->EnumerateAdapters(&options);
    if (adapters.empty()) {
        LOGE("No WebGPU adapters found");
        return nullptr;
    }
    dawn::native::Adapter adapter = adapters[0];

    WGPUDeviceDescriptor deviceDesc = {};
    WGPUUncapturedErrorCallbackInfo errCb = {};
    errCb.callback = PrintDeviceError;
    deviceDesc.uncapturedErrorCallbackInfo = errCb;

    WGPUDevice cDevice = adapter.CreateDevice(&deviceDesc);
    g_device = wgpu::Device::Acquire(cDevice);
    g_queue = g_device.GetQueue();

    return g_device.Get();
}

API_EXPORT void* webgpu_rend_create_texture(int32_t width, int32_t height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_device) return nullptr;
    try {
        auto tex = std::make_unique<AndroidTextureObject>(width, height);
        void* ptr = tex.get();
        g_textures[ptr] = std::move(tex);
        return ptr;
    } catch (std::exception& e) {
        LOGE("Failed to create texture: %s", e.what());
        return nullptr;
    }
}

API_EXPORT int64_t webgpu_rend_get_texture_id(void* t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_textures.count(t) ? g_textures[t]->flutter_texture_id : -1;
}

API_EXPORT void* webgpu_rend_get_wgpu_texture(void* t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_textures.count(t) ? g_textures[t]->working_texture.Get() : nullptr;
}

API_EXPORT void* webgpu_rend_get_wgpu_texture_view(void* t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_textures.count(t) ? g_textures[t]->working_view.Get() : nullptr;
}

API_EXPORT void webgpu_rend_texture_begin_access(void* t) {}
API_EXPORT void webgpu_rend_texture_end_access(void* t) {}

API_EXPORT void webgpu_rend_present_texture(void* t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_textures.count(t)) return;
    auto& obj = g_textures[t];

    wgpu::SurfaceTexture surfaceTexture;
    obj->surface.GetCurrentTexture(&surfaceTexture);

    if (surfaceTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        surfaceTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        LOGE("Failed to get surface texture status: %d", (int)surfaceTexture.status);
        return;
    }

    wgpu::CommandEncoder encoder = g_device.CreateCommandEncoder();

    wgpu::TexelCopyTextureInfo src = {};
    src.texture = obj->working_texture;

    wgpu::TexelCopyTextureInfo dst = {};
    dst.texture = surfaceTexture.texture;

    wgpu::Extent3D copySize = {(uint32_t)obj->width, (uint32_t)obj->height, 1};

    encoder.CopyTextureToTexture(&src, &dst, &copySize);

    wgpu::CommandBuffer cmd = encoder.Finish();
    g_queue.Submit(1, &cmd);

    obj->surface.Present();
}

API_EXPORT void webgpu_rend_dispose_texture(void* t) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_textures.erase(t);
}

}  // extern "C"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_vm = vm;
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    jclass clazz = env->FindClass("com/funguscow/webgpu_rend/WebgpuRendPlugin");
    g_plugin_class = (jclass)env->NewGlobalRef(clazz);

    g_create_texture_mid = env->GetStaticMethodID(clazz, "createTexture", "(II)I");
    g_get_id_mid = env->GetStaticMethodID(clazz, "getTextureId", "(I)J");
    g_get_surface_mid = env->GetStaticMethodID(clazz, "getSurface", "(I)Landroid/view/Surface;");
    g_dispose_mid = env->GetStaticMethodID(clazz, "disposeTexture", "(I)V");

    return JNI_VERSION_1_6;
}