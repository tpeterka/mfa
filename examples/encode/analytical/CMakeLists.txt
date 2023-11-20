include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (analytical                             analytical.cpp)
add_executable              (multiblock_noblend_continuous          multiblock_noblend_continuous.cpp)
add_executable              (multiblock_blend_continuous            multiblock_blend_continuous.cpp)

target_link_libraries       (analytical                             ${libraries})
target_link_libraries       (multiblock_noblend_continuous          ${libraries})
target_link_libraries       (multiblock_blend_continuous            ${libraries})

set_property                (TARGET analytical                      PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET multiblock_noblend_continuous   PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET multiblock_blend_continuous     PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS analytical multiblock_noblend_continuous multiblock_blend_continuous
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/encode/analytical
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)