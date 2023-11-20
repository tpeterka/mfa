include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (analytical                             analytical.cpp)
add_executable              (analytical_multiblock                  analytical_multiblock.cpp)
add_executable              (analytical_multiblock_blend            analytical_multiblock_blend.cpp)

target_link_libraries       (analytical                             ${libraries})
target_link_libraries       (analytical_multiblock                  ${libraries})
target_link_libraries       (analytical_multiblock_blend            ${libraries})

set_property                (TARGET analytical                      PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET analytical_multiblock           PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET analytical_multiblock_blend     PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS analytical analytical_multiblock analytical_multiblock_blend
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/encode/analytical
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)