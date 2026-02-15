package com.funguscow.webgpu_rend

import android.view.Surface
import androidx.annotation.Keep
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.view.TextureRegistry
import java.util.concurrent.ConcurrentHashMap

class WebgpuRendPlugin : FlutterPlugin {

  // C++ side calls these via JNI
  @Keep
  companion object {
    private var textureRegistry: TextureRegistry? = null
    private val producers = ConcurrentHashMap<Int, TextureRegistry.SurfaceProducer>()
    private var nextHandle = 1

    @JvmStatic
    @Keep
    fun createTexture(width: Int, height: Int): Int {
      val reg = textureRegistry ?: return -1
      val producer = reg.createSurfaceProducer()
      producer.setSize(width, height)
      val handle = nextHandle++
      producers[handle] = producer
      return handle
    }

    @JvmStatic
    @Keep
    fun getTextureId(handle: Int): Long {
      return producers[handle]?.id() ?: -1L
    }

    @JvmStatic
    @Keep
    fun getSurface(handle: Int): Surface? {
      return producers[handle]?.getSurface()
    }

    @JvmStatic
    @Keep
    fun disposeTexture(handle: Int) {
      producers.remove(handle)?.release()
    }
  }

  override fun onAttachedToEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
    System.loadLibrary("webgpu_rend_android")
    textureRegistry = binding.textureRegistry
  }

  override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
    textureRegistry = null
    producers.values.forEach { it.release() }
    producers.clear()
  }
}
