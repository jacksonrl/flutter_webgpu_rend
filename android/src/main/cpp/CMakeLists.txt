cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "webgpu_rend")
project(${PROJECT_NAME} LANGUAGES CXX)

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")

# version info
set(DAWN_RELEASE_TAG "v20260121.191546")
set(DAWN_COMMIT_SHA "35d0f11eaa71ef2673dbedb376ceb2a2ad89f222") 

set(DAWN_DIR "${ROOT_DIR}/third_party/dawn")

if(NOT EXISTS "${DAWN_DIR}/include/webgpu/webgpu.h" OR NOT EXISTS "${DAWN_DIR}/lib/android/${ANDROID_ABI}/libwebgpu_dawn.a")
    message(STATUS "Dawn dependencies missing. Preparing to download...")

    file(MAKE_DIRECTORY "${DAWN_DIR}")
    set(TEMP_DIR "${DAWN_DIR}/temp_extract")
    file(MAKE_DIRECTORY "${TEMP_DIR}")

    if(NOT EXISTS "${DAWN_DIR}/include/webgpu/webgpu.h")
        set(HEADERS_FILENAME "dawn-headers-${DAWN_COMMIT_SHA}.tar.gz")
        set(HEADERS_URL "https://github.com/google/dawn/releases/download/${DAWN_RELEASE_TAG}/${HEADERS_FILENAME}")
        
        message(STATUS "Downloading Headers: ${HEADERS_URL}")
        file(DOWNLOAD ${HEADERS_URL} "${TEMP_DIR}/${HEADERS_FILENAME}" SHOW_PROGRESS STATUS DOWNLOAD_STATUS)
        
        list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
        if(NOT STATUS_CODE EQUAL 0)
            message(FATAL_ERROR "Headers Download failed: ${DOWNLOAD_STATUS}")
        endif()

        message(STATUS "Extracting Headers...")
        file(ARCHIVE_EXTRACT INPUT "${TEMP_DIR}/${HEADERS_FILENAME}" DESTINATION "${TEMP_DIR}")
        
        file(RENAME "${TEMP_DIR}/dawn-headers/include" "${DAWN_DIR}/include")
    endif()

    if(NOT EXISTS "${DAWN_DIR}/lib/android")
        set(ANDROID_FILENAME "dawn-android-${DAWN_COMMIT_SHA}.tar.gz")
        set(ANDROID_URL "https://github.com/google/dawn/releases/download/${DAWN_RELEASE_TAG}/${ANDROID_FILENAME}")

        message(STATUS "Downloading Android Libs: ${ANDROID_URL}")
        file(DOWNLOAD ${ANDROID_URL} "${TEMP_DIR}/${ANDROID_FILENAME}" SHOW_PROGRESS STATUS DOWNLOAD_STATUS)

        list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
        if(NOT STATUS_CODE EQUAL 0)
            message(FATAL_ERROR "Android Lib Download failed: ${DOWNLOAD_STATUS}")
        endif()

        message(STATUS "Extracting Android Libs...")
        file(ARCHIVE_EXTRACT INPUT "${TEMP_DIR}/${ANDROID_FILENAME}" DESTINATION "${TEMP_DIR}")

        file(MAKE_DIRECTORY "${DAWN_DIR}/lib")
        
        file(RENAME "${TEMP_DIR}/dawn-android-${DAWN_COMMIT_SHA}" "${DAWN_DIR}/lib/android")
    endif()

    # Cleanup
    file(REMOVE_RECURSE "${TEMP_DIR}")
    message(STATUS "Dawn Setup Complete.")
else()
    message(STATUS "Dawn found at ${DAWN_DIR}")
endif()

if("${ANDROID_ABI}" STREQUAL "x86" OR "${ANDROID_ABI}" STREQUAL "x86_64")
    message(WARNING "WebGPU Rend: x86/x86_64 architecture skipped (ensure checks match available libs).")
    return() 
endif()

set(DAWN_LIB_PATH ${DAWN_DIR}/lib/android/${ANDROID_ABI}/libwebgpu_dawn.a)

if(NOT EXISTS ${DAWN_LIB_PATH})
    message(FATAL_ERROR "Dawn library not found for ABI ${ANDROID_ABI} at path: ${DAWN_LIB_PATH}")
endif()

add_library(webgpu_rend_android SHARED webgpu_rend_android_api.cpp)

target_include_directories(webgpu_rend_android PRIVATE
    ${ROOT_DIR}/src
    ${DAWN_DIR}/include
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(webgpu_rend_android PRIVATE
    ${log-lib}
    ${android-lib}
    ${DAWN_LIB_PATH}
)