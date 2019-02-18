include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (write_vtk              write_vtk.cpp writer.cpp)

target_link_libraries       (write_vtk              ${libraries})

set_property                (TARGET write_vtk               PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS write_vtk
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/convert/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
