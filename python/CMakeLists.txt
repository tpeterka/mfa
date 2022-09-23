if                          (CMAKE_LIBRARY_OUTPUT_DIRECTORY)        # to work with pip
    set                     (MODULE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
else                        ()
    set                     (MODULE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif                       ()

list                        (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package                (MPI4Py)

# Copy mfa/*.py
file                        (MAKE_DIRECTORY "${MODULE_OUTPUT_DIRECTORY}/mfa")
file                        (GLOB PYMFA_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/mfa/*.py")
add_custom_target           (pymfa ALL ${CMAKE_COMMAND} -E copy_directory
                                    ${CMAKE_CURRENT_SOURCE_DIR}/mfa ${MODULE_OUTPUT_DIRECTORY}/mfa
                                    DEPENDS ${PYMFA_PYTHON})

# Build mfa/_mfa.so
set                         (MFA_SOURCES
                                src/mfa.cpp
                                src/block.cpp)

pybind11_add_module         (_mfa ${MFA_SOURCES})
target_link_libraries       (_mfa PRIVATE ${libraries})
set_target_properties       (_mfa PROPERTIES OUTPUT_NAME mfa/_mfa)
set_target_properties       (_mfa PROPERTIES CXX_VISIBILITY_PRESET "default")

if                          (MPI4Py_FOUND)
    target_compile_definitions  (_mfa PUBLIC "-DMFA_MPI4PY")
    target_include_directories  (_mfa PUBLIC ${MPI4Py_INCLUDE_DIR})
endif()

add_subdirectory            (examples)
