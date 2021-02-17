project                     (mfa)
cmake_minimum_required      (VERSION 3.1)

option                      (wrapped_mpi    "MPI compiler wrapper requires no further MPI libs"     OFF     )
option                      (mfa_thread     "tbb, kokkos, sycl, or serial"                          serial  )
option                      (s3d_infile     "Path/filename of s3d dataset (for unit testing)"               )
option                      (python         "Build Python bindings"                                 OFF     )

# C++11
set                         (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

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
 if                         (NOT wrapped_mpi)
     find_package           (MPI REQUIRED)
     set                    (libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
 endif                      ()

# MFA threading model
if                          (mfa_thread MATCHES "tbb")
    message                 ("Using TBB threading")
    add_definitions         (-DMFA_TBB)
    find_package            (TBB QUIET)
    find_path               (TBB_INCLUDE_DIR        tbb/tbb.h)
    find_library            (TBB_LIBRARY NAMES      tbb)
    if                      (TBB_INCLUDE_DIR AND TBB_LIBRARY)
        include_directories ("${TBB_INCLUDE_DIR}")
        set                 (libraries ${libraries} ${TBB_LIBRARY})
        message             ("Found TBB in ${TBB_INCLUDE_DIR} and in ${TBB_LIBRARY}")
    else                    (TBB_INCLUDE_DIR AND TBB_LIBRARY)
        message             (FATAL_ERROR "Could not find TBB")
    endif                   ()
elseif                      (mfa_thread MATCHES "kokkos")
    message                 ("Using Kokkos threading")
    add_definitions         (-DMFA_KOKKOS)
    # TODO: for cuda only; how to check for compiler==nvcc_wrapper?
    set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-relaxed-constexpr")
    set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")
    include_directories     (${KOKKOS_INCLUDE_DIRS})
    set                     (libraries ${libraries} ${KOKKOS_LIBRARY} ${CMAKE_DL_LIBS})
elseif                      (mfa_thread MATCHES "sycl")
    message                 ("Using SYCL threading")
    add_definitions         (-DMFA_SYCL)
    set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set                     (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lsycl -lOpenCL")
else                        ()
    message                 ("Using no threading (serial)")
    add_definitions         (-DMFA_SERIAL)
endif                       ()

# Threads
find_package                (Threads)

# DIY
find_path                   (DIY_INCLUDE_DIRS diy/types.h)

# Coin-OR CLP
find_path                   (CLP_INCLUDE_DIRS   ClpSimplex.hpp)
find_library                (CLP_LIB NAMES      Clp)
if                          (NOT CLP_LIB OR NOT CLP_INCLUDE_DIRS)
    message                 ("Coin-OR CLP not found; weights are disabled")
    set                     (CLP_INCLUDE_DIRS   "")
    set                     (CLP_LIB            "")
    set                     (COIN_UTILS_LIB     "")
    add_definitions         (-DMFA_NO_WEIGHTS)
else                        ()
    message                 ("Coin-OR CLP found; weights are enabled")
endif                       ()

# Include dirs
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/cppoptlib/include
                            ${DIY_INCLUDE_DIRS}
                            ${EIGEN_INCLUDE_DIRS}
                            SYSTEM
                            ${MPI_INCLUDE_PATH}
                            )
if                          (NOT "${CLP_INCLUDE_DIRS}" STREQUAL "")
    include_directories     (${CLP_INCLUDE_DIRS})
endif                       ()
if                          (NOT "${Boost_INCLUDE_DIRS}" STREQUAL "")
    include_directories     (${Boost_INCLUDE_DIRS})
endif                       ()

# Libraries
set                         (libraries
                             ${libraries}
                             ${CLP_LIB}
                             ${COIN_UTILS_LIB}
                             ${CMAKE_THREAD_LIBS_INIT})

# Testing
# NB testing must be enabled before adding tests subdirectory, otherwise no tests will be found
enable_testing              ()
include                     (CTest)

# Subdirectories
if                          (DIY_INCLUDE_DIRS AND EIGEN_INCLUDE_DIRS)
    add_subdirectory        (examples)
    add_subdirectory        (tests)
else                        ()
    message                 (ERROR "Need diy2 and eigen in order to build mfa")
endif                       ()
if                          (python)
    add_subdirectory        (include/pybind11)
    add_subdirectory        (python)
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

