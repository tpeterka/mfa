include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (nasa-vis                               nasa-vis.cpp)

target_link_libraries       (nasa-vis                               ${libraries})

set_property                (TARGET nasa-vis                        PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS nasa-vis
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/fixed/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
