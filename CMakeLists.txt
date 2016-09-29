project                     (mfa)
cmake_minimum_required      (VERSION 2.8)

option                      (bgq            "Build on BG/Q"                        OFF)
option                      (diy_thread     "Enable diy threading"                 OFF)

if                          (bgq)
    add_definitions         (-DBGQ)
endif                       (bgq)

# OSX flags
if                          (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions	    (-DMAC_OSX)
endif                       (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

 # MPI
find_package                (MPI REQUIRED)
if                          (NOT bgq)
    set                     (libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif                       (NOT bgq)

# Threads
find_package		    (Threads)
find_package		    (OpenMP)
if 			    (OPENMP_FOUND)
  set                       (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set                       (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else			    ()
  message		    ("OpenMP not found")
  add_definitions	    (-DMFA_NO_OPENMP)
endif                       ()
if                          (NOT diy_thread)
    message                 ("Diy threading is disabled; setting diy threads will have no effect")
    add_definitions         (-DDIY_NO_THREADS)
endif                       (NOT diy_thread)

# DIY
find_path                   (DIY_INCLUDE_DIRS diy/types.h)

# Include dirs
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include
			     ${DIY_INCLUDE_DIRS}
			     ${EIGEN_INCLUDE_DIRS}
                             SYSTEM ${MPI_INCLUDE_PATH}
                            )

# Libraries
set			    (libraries
                             ${libraries}
                             ${CMAKE_THREAD_LIBS_INIT})

# Subdirectories
if                          (DIY_INCLUDE_DIRS AND EIGEN_INCLUDE_DIRS)
    add_subdirectory        (src)
    add_subdirectory        (examples)
else                        ()
    message                 (ERROR "Need diy2 and eigen in order to build mfa")
endif                       ()

# Install the headers
file(GLOB DEPLOY_FILES_AND_DIRS "${PROJECT_SOURCE_DIR}/include/*")
foreach(ITEM ${DEPLOY_FILES_AND_DIRS})
   if( IS_DIRECTORY "${ITEM}" )
      list( APPEND DIRS_TO_DEPLOY "${ITEM}" )
   else()
      list( APPEND FILES_TO_DEPLOY "${ITEM}" )
   endif()
endforeach()
install( FILES ${FILES_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
install( DIRECTORY ${DIRS_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
