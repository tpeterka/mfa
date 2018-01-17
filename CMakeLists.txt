project                     (mfa)
cmake_minimum_required      (VERSION 2.8)

option                      (bgq            "Build on BG/Q"                        OFF)
option                      (diy_thread     "Enable diy threading"                 OFF)

if                          (bgq)
    add_definitions         (-DBGQ)
endif                       (bgq)

# C++11
set                         (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# OSX flags
if                          (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_definitions           (-DMAC_OSX)
  set                       (CMAKE_MACOSX_RPATH on)

    # --- following RPATH settings are for Sierra w/ Clang, hopefully they don't hurt other versions
    # ref: https://cmake.org/Wiki/CMake_RPATH_handling
    # use, i.e. don't skip, the full RPATH for the build tree
    set                     (CMAKE_SKIP_BUILD_RPATH            false)
    # when building, don't use the install RPATH already (but later on when installing)
    set                     (CMAKE_BUILD_WITH_INSTALL_RPATH    false)
    # set RPATH to install path
    set                     (CMAKE_INSTALL_RPATH               "${CMAKE_INSTALL_PREFIX}/lib")
    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set                     (CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
    # the RPATH to be used when installing, but only if it's not a system directory
    list                    (FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
                                                               "${CMAKE_INSTALL_PREFIX}/lib"
                                                               isSystemDir)
    if                      ("${isSystemDir}" STREQUAL         "-1")
      set                   (CMAKE_INSTALL_RPATH               "${CMAKE_INSTALL_PREFIX}/lib")
    endif                   ()
endif                       (${CMAKE_SYSTEM_NAME} MATCHES      "Darwin")

 # MPI
find_package                (MPI REQUIRED)
if                          (NOT bgq)
    set                     (libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif                       (NOT bgq)

# TBB
find_package                (TBB QUIET)
find_path                   (TBB_INCLUDE_DIR        tbb.h)
find_library                (TBB_LIBRARY NAMES      tbb)
if                          (TBB_INCLUDE_DIR AND TBB_LIBRARY)
  include_directories       ("${TBB_INCLUDE_DIR}")
  set                       (libraries ${TBB_LIBRARY})
  message                   (STATUS "Found TBB in ${TBB_INCLUDE_DIR} and in ${TBB_LIBRARY}")
else                        (TBB_INCLUDE_DIR AND TBB_LIBRARY)
  message                   (ERROR "Could not find TBB")
endif                       (TBB_INCLUDE_DIR AND TBB_LIBRARY)

# Threads
find_package                (Threads)
find_package                (OpenMP)
if                          (OPENMP_FOUND)
  set                       (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set                       (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else                        ()
  message                   ("OpenMP not found")
  add_definitions           (-DMFA_NO_OPENMP)
endif                       ()
if                          (NOT diy_thread)
    message                 ("Diy threading is disabled; setting diy threads will have no effect")
    add_definitions         (-DDIY_NO_THREADS)
endif                       (NOT diy_thread)

# DIY
find_path                   (DIY_INCLUDE_DIRS diy/types.h)

# Coin-OR CLP
find_path                   (CLP_INCLUDE_DIRS   ClpSimplex.hpp)
find_library                (CLP_LIBS NAMES     Clp)

# Include dirs
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${DIY_INCLUDE_DIRS}
                            ${EIGEN_INCLUDE_DIRS}
                            SYSTEM
                            ${CLP_INCLUDE_DIRS}
                            ${MPI_INCLUDE_PATH}
                            )

# Libraries
set                         (libraries
                             ${libraries}
                             ${CLP_LIB}
                             ${COIN_UTILS_LIB}
                             ${CMAKE_THREAD_LIBS_INIT})

# Subdirectories
if                          (DIY_INCLUDE_DIRS AND EIGEN_INCLUDE_DIRS)
    add_subdirectory        (src)
    add_subdirectory        (examples)
else                        ()
    message                 (ERROR "Need diy2 and eigen in order to build mfa")
endif                       ()

# Install the headers
file                        (GLOB DEPLOY_FILES_AND_DIRS "${PROJECT_SOURCE_DIR}/include/*")
foreach                     (ITEM ${DEPLOY_FILES_AND_DIRS})
   if                       (IS_DIRECTORY "${ITEM}")
      list                  (APPEND DIRS_TO_DEPLOY "${ITEM}")
   else                     ()
      list                  (APPEND FILES_TO_DEPLOY "${ITEM}")
   endif                    ()
endforeach                  ()
install                     (FILES ${FILES_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install                     (DIRECTORY ${DIRS_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
