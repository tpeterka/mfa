include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (vol_iterator           vol_iterator.cpp)

target_link_libraries       (vol_iterator           ${libraries})

set_property                (TARGET vol_iterator    PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS vol_iterator
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/vol_iterator/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
