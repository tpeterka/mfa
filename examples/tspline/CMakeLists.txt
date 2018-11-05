include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (tmesh-test             tmesh-test.cpp)
target_link_libraries       (tmesh-test             ${libraries})

set_property                (TARGET  tmesh-test     PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install                     (TARGETS tmesh-test
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/tspline/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
