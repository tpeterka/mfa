cmake_minimum_required      (VERSION 3.15)
cmake_policy                (SET CMP0048 NEW) # Tell old versions of CMake to treat project version "correctly"

project                     (mfa)

option                      (wrapped_mpi    "MPI compiler wrapper requires no further MPI libs, true/false (default false)" OFF     )
option                      (mfa_thread     "tbb, kokkos, sycl, or serial (default serial)"                                 serial  )
option                      (s3d_infile     "Path/filename of s3d dataset (for unit testing)"                                       )
option                      (mfa_build_examples "Build MFA examples (default true)"                                         ON      )
option                      (mfa_build_tests "Build MFA tests (default true)"                                               ON      )
option                      (mfa_python     "Build Python bindings, true/false (default false)"                             OFF     )
option                      (eigen_thread   "Use OpenMP for Eigen threading, true/false (default false)"                    OFF     )
option                      (KOKKOS_INCLUDE_DIRS    "Path of include directories for Kokkos (optional)"                             )
option                      (KOKKOS_LIBRARY "Kokkos library (optional)"                                                             )
option                      (OPENMP_LIBRARY "OpenMP library if not found automatically (optional)"                                  )

# C++17
set                         (CMAKE_CXX_STANDARD 17)
set                         (CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

# MPI
if                          (NOT wrapped_mpi)
    find_package            (MPI REQUIRED)
    set                     (libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif                       ()

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
   # set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
   # set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-relaxed-constexpr")
   # set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")
   # include_directories     (${KOKKOS_INCLUDE_DIRS})
   # set                     (libraries ${libraries} ${KOKKOS_LIBRARY} ${CMAKE_DL_LIBS})
    find_package ( Kokkos REQUIRED )
    message                 ( " found KOKKOS in "  ${Kokkos_DIR} )
    set                     ( KK_LIB  "Kokkos::kokkos" ) 
elseif                      (mfa_thread MATCHES "sycl")
    message                 ("Using SYCL threading")
    add_definitions         (-DMFA_SYCL)
    set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set                     (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lsycl -lOpenCL")
else                        ()
    message                 ("Using no threading (serial)")
    add_definitions         (-DMFA_SERIAL)
endif                       ()

# Eigen threading (requires OpenMP to be available)
if                          (eigen_thread)
    find_package            (OpenMP)
    message                 ("Using OpenMP for Eigen")
    add_definitions         (-DEIGEN_OPENMP)
    if                      (APPLE)
        set                 (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
        if                  (${OPENMP_CXX_FOUND})
            set             (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
        elseif              (DEFINED OPENMP_INCLUDE_DIRS AND DEFINED OPENMP_LIBRARY)
            include_directories (${OPENMP_INCLUDE_DIRS})
            set             (libraries ${libraries} ${OPENMP_LIBRARY})
            message         ("Using user-provided OpenMP headers and library: ${OPENMP_INCLUDE_DIRS} ${OPENMP_LIBRARY}")
        else                ()
            message         ("Error: cannot find OpenMP and OPENMP_INCLUDE_DIRS or OPENMP_LIBRARY is not set")
            message         (FATAL_ERROR "Either provide OPENMP_INCLUDE_DIRS and OPENMP_LIBRARY explicitly or disable eigen_thread")
        endif               ()
    else                    ()
        set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif                   ()
endif                       ()

# Threads
find_package                (Threads)

# DIY
option                      (build_examples "" OFF)
option                      (build_tests "" OFF)
add_subdirectory            (include/diy)
set                         (libraries ${libraries} diy)

# fmt
option                      (FMT_INSTALL "" OFF)
add_subdirectory            (include/fmt)
set                         (libraries ${libraries} fmt::fmt)

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
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/diy/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/fmt/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/eigen3
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
                             ${CMAKE_DL_LIBS} 
                             ${CLP_LIB}
                             ${COIN_UTILS_LIB}
                             ${CMAKE_THREAD_LIBS_INIT}
                             ${KK_LIB} )



if                          (mfa_build_examples)
    add_subdirectory        (examples)
endif                       (mfa_build_examples)

# Build Tests
# NB testing must be enabled before adding tests subdirectory, otherwise no tests will be found
if                          (mfa_build_tests)
    enable_testing          ()
    include                 (CTest)
    add_subdirectory        (tests)
endif                       (mfa_build_tests)

# Python
if                          (mfa_python)
find_package                (Python COMPONENTS Interpreter Development)
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

