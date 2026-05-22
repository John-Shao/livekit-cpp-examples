# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# board_loopback — RV1126B port application.
#
# Single-process bidirectional meeting client for the ATK-DLRV1126B board:
# V4L2 camera capture + MPP hardware codec + DRM/KMS rendering + ALSA audio
# + standalone AEC + HTTP control API + split-screen UI.
#
# Linux-only: hard-depends on ALSA (ES8389 codec), libdrm (KMS plane) and the
# Linux V4L2 / NEON headers. Skipped on non-Linux hosts so the rest of the
# example collection still builds there.
if(NOT (UNIX AND NOT APPLE))
  message(STATUS "board_loopback: skipped (Linux-only example)")
  return()
endif()

add_executable(BoardLoopback
  main.cpp
)

# ${CMAKE_CURRENT_SOURCE_DIR} carries the vendored header-only cpp-httplib
# (httplib.h, MIT licensed) that main.cpp includes as <httplib.h>.
target_include_directories(BoardLoopback PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ALSA for ES8389 codec capture/playback + libdrm for KMS plane rendering.
# Linked unconditionally; the cross sysroot must provide libasound and libdrm.
target_link_libraries(BoardLoopback PRIVATE ${LIVEKIT_CORE_TARGET} asound drm)

# -rdynamic exports dynamic symbols so the in-app crash handler's backtrace()
# can resolve frames; -g keeps source-line info. Both are Release-safe.
target_link_options(BoardLoopback PRIVATE -rdynamic)
target_compile_options(BoardLoopback PRIVATE -g)

# libdrm headers live under <sysroot>/usr/include/{libdrm,drm} in Buildroot.
find_path(LIBDRM_INCLUDE_DIR drm.h PATH_SUFFIXES libdrm drm)
if(LIBDRM_INCLUDE_DIR)
  target_include_directories(BoardLoopback PRIVATE ${LIBDRM_INCLUDE_DIR})
endif()
