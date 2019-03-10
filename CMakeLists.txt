cmake_minimum_required(VERSION 2.8)
project(final)

set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig REQUIRED)

pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})
message(STATUS "GLFW included at ${GLFW_INCLUDE_DIRS}")

pkg_search_module(GLM REQUIRED glm)
include_directories(${GLM_INCLUDE_DIRS})
message(STATUS "GLM included at ${GLM_INCLUDE_DIRS}")

pkg_search_module(ASSIMP REQUIRED assimp)
include_directories(${ASSIMP_INCLUDE_DIRS})
message(STATUS "ASSIMP included at ${ASSIMP_INCLUDE_DIRS}")

if (APPLE)
    INCLUDE_DIRECTORIES(/System/Library/Frameworks)

    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    message(STATUS "Cocoa included at ${COCOA_LIBRARY}")

    FIND_LIBRARY(OpenGL_LIBRARY OpenGL)
    message(STATUS "OpenGL included at ${OpenGL_LIBRARY}")

    FIND_LIBRARY(IOKit_LIBRARY IOKit)
    message(STATUS "IOKit included at ${IOKit_LIBRARY}")

    FIND_LIBRARY(CoreVideo_LIBRARY CoreVideo)
    message(STATUS "CoreVideo included at ${CoreVideo_LIBRARY}")

    MARK_AS_ADVANCED(COCOA_LIBRARY OpenGL_LIBRARY)
    SET(APPLE_LIBS ${COCOA_LIBRARY} ${IOKit_LIBRARY} ${OpenGL_LIBRARY} ${CoreVideo_LIBRARY})
    SET(APPLE_LIBS ${APPLE_LIBS} ${GLFW_LIBRARIES} ${GLM_LIBRARIES} ${ASSIMP_LIBRARIES})
    set(LIBS ${LIBS} ${APPLE_LIBS})
endif(APPLE)

configure_file(configuration/root_directory.h.in configuration/root_directory.h)
include_directories(${CMAKE_BINARY_DIR}/configuration)

# add STB
add_library(STB_IMAGE "src/stb_image.cpp")
set(LIBS ${LIBS} STB_IMAGE)

# add GLAD
add_library(GLAD "src/glad.c")
set(LIBS ${LIBS} GLAD)

file(GLOB SHADERS
        "shaders/*.frag"
        "shaders/*.vert"
        )

macro(makeLink src dest target)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest}  DEPENDS  ${dest} COMMENT "mklink ${src} -> ${dest}")
endmacro()

add_executable(final main.cpp)
target_link_libraries(final ${LIBS})

foreach(SHADER ${SHADERS})
    if(APPLE)
        set_target_properties(final PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin/")
        set_target_properties(final PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/bin/")
        set_target_properties(final PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/bin/")

        # create symbolic link for *.vs *.fs *.gs
        get_filename_component(SHADERNAME ${SHADER} NAME)
        makeLink(${SHADER} ${CMAKE_CURRENT_BINARY_DIR}/bin/${SHADERNAME} final)
    endif(APPLE)
endforeach(SHADER)


include_directories(${CMAKE_SOURCE_DIR}/includes)
