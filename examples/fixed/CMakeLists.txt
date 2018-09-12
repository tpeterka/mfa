include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (fixed                      fixed.cpp)
add_executable              (fixed_multiblock           fixed_multiblock.cpp)
add_executable              (fixed_multiblock_ghost     fixed_multiblock_ghost.cpp)
target_link_libraries       (fixed                      ${libraries})
target_link_libraries       (fixed_multiblock           ${libraries})
target_link_libraries       (fixed_multiblock_ghost     ${libraries})

set_property                (TARGET fixed                   PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET fixed_multiblock        PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET fixed_multiblock_ghost  PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install                     (TARGETS fixed fixed_multiblock fixed_multiblock_ghost
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/fixed/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
