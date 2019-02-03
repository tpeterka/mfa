include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (adaptive       adaptive.cpp)
add_executable              (hep_adaptive   hep_adaptive.cpp)

target_link_libraries       (adaptive       ${libraries})
target_link_libraries       (hep_adaptive   ${libraries})

set_property                (TARGET  adaptive       PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET  hep_adaptive   PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS adaptive hep_adaptive
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/adaptive/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
