include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (gridded_2d                             gridded_2d.cpp)
add_executable              (gridded_3d                             gridded_3d.cpp)
add_executable              (grid_multiblock_blend                  grid_multiblock_blend.cpp)

target_link_libraries       (gridded_2d                             ${libraries})
target_link_libraries       (gridded_3d                             ${libraries})
target_link_libraries       (grid_multiblock_blend                  ${libraries})

set_property                (TARGET gridded_2d                      PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET gridded_3d                      PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET grid_multiblock_blend           PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS gridded_2d gridded_3d grid_multiblock_blend
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/encode/grid
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)