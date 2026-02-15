cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "webgpu_rend")
project(${PROJECT_NAME} LANGUAGES CXX)

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# version info
set(DAWN_RELEASE_TAG "v20260121.191546")
set(DAWN_COMMIT_SHA "35d0f11eaa71ef2673dbedb376ceb2a2ad89f222")

set(DAWN_DIR "${ROOT_DIR}/third_party/dawn")

if(NOT EXISTS "${DAWN_DIR}/include/webgpu/webgpu.h" OR NOT EXISTS "${DAWN_DIR}/lib/windows/Release/webgpu_dawn.lib")
    message(STATUS "Dawn libraries not found. Downloading from GitHub...")
    
    file(MAKE_DIRECTORY "${DAWN_DIR}")
    set(TEMP_DIR "${DAWN_DIR}/temp_extract")
    file(MAKE_DIRECTORY "${TEMP_DIR}")

    function(setup_dawn_windows TYPE DEST_DIR_NAME)
        set(ARCHIVE_NAME "Dawn-${DAWN_COMMIT_SHA}-windows-latest-${TYPE}.tar.gz")
        set(DOWNLOAD_URL "https://github.com/google/dawn/releases/download/${DAWN_RELEASE_TAG}/${ARCHIVE_NAME}")
        set(LOCAL_ZIP "${TEMP_DIR}/${ARCHIVE_NAME}")

        message(STATUS "Downloading Windows ${TYPE} artifact...")
        file(DOWNLOAD ${DOWNLOAD_URL} ${LOCAL_ZIP} SHOW_PROGRESS STATUS DOWNLOAD_STATUS)

        list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
        if(NOT STATUS_CODE EQUAL 0)
            message(FATAL_ERROR "Download failed for ${TYPE}: ${DOWNLOAD_STATUS}")
        endif()

        message(STATUS "Extracting ${TYPE}...")
        file(MAKE_DIRECTORY "${TEMP_DIR}/${TYPE}")
        file(ARCHIVE_EXTRACT INPUT ${LOCAL_ZIP} DESTINATION "${TEMP_DIR}/${TYPE}")

        set(EXTRACTED_ROOT "${TEMP_DIR}/${TYPE}/Dawn-${DAWN_COMMIT_SHA}-windows-latest-${TYPE}")

        file(MAKE_DIRECTORY "${DAWN_DIR}/lib/windows/${DEST_DIR_NAME}")
        file(COPY "${EXTRACTED_ROOT}/lib/webgpu_dawn.lib" DESTINATION "${DAWN_DIR}/lib/windows/${DEST_DIR_NAME}")

        if("${TYPE}" STREQUAL "Release")
            if(NOT EXISTS "${DAWN_DIR}/include")
                message(STATUS "Installing Headers...")
                file(MAKE_DIRECTORY "${DAWN_DIR}/include")
                file(COPY "${EXTRACTED_ROOT}/include/" DESTINATION "${DAWN_DIR}/include")
            endif()
        endif()
    endfunction()

    setup_dawn_windows("Debug" "Debug")
    setup_dawn_windows("Release" "Release")

    file(REMOVE_RECURSE "${TEMP_DIR}")
    message(STATUS "Dawn Setup Complete.")
else()
    message(STATUS "Dawn found at ${DAWN_DIR}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PLUGIN_NAME "webgpu_rend_plugin")

list(APPEND PLUGIN_SOURCES
  "webgpu_rend_plugin.cpp"
  "webgpu_rend_plugin.h"
)

add_library(${PLUGIN_NAME} SHARED
  "include/webgpu_rend/webgpu_rend_plugin_c_api.h"
  "webgpu_rend_plugin_c_api.cpp"
  ${PLUGIN_SOURCES}
)

apply_standard_settings(${PLUGIN_NAME})

target_compile_definitions(${PLUGIN_NAME} PRIVATE 
  FLUTTER_PLUGIN_IMPL
  _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
)

set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
  
target_include_directories(${PLUGIN_NAME} PRIVATE
  "${DAWN_DIR}/include"
)
  
target_link_libraries(${PLUGIN_NAME} PRIVATE
  flutter
  flutter_wrapper_plugin
  d3d11
  d3d12 
  dxgi
  dxguid
  dxcompiler.lib
  Kernel32.lib
  mincore.lib
  debug "${DAWN_DIR}/lib/windows/Debug/webgpu_dawn.lib"
  optimized "${DAWN_DIR}/lib/windows/Release/webgpu_dawn.lib"
)

# Ignores PDB not found warnings for the static Dawn lib
target_link_options(${PLUGIN_NAME} PRIVATE "/ignore:4099")

# You might need to adjust this version number to match your installed SDK.
set(WINDOWS_SDK_BIN_DIR "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64")

if(NOT EXISTS "${WINDOWS_SDK_BIN_DIR}/dxcompiler.dll")
    message(FATAL_ERROR "dxcompiler.dll not found at '${WINDOWS_SDK_BIN_DIR}'. "
                        "Please update the WINDOWS_SDK_BIN_DIR path in your "
                        "windows/CMakeLists.txt to match your installed Windows SDK version.")
endif()

set(webgpu_rend_bundled_libraries
  "${WINDOWS_SDK_BIN_DIR}/dxcompiler.dll"
  "${WINDOWS_SDK_BIN_DIR}/dxil.dll"
  "${WINDOWS_SDK_BIN_DIR}/d3dcompiler_47.dll"
  PARENT_SCOPE
)