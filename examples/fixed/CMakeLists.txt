include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (fixed                                  fixed.cpp)
add_executable              (multiblock_noblend_continuous          multiblock_noblend_continuous.cpp)
add_executable              (multiblock_blend_continuous            multiblock_blend_continuous.cpp)
add_executable              (multiblock_blend_discrete              multiblock_blend_discrete.cpp)

target_link_libraries       (fixed                                  ${libraries})
target_link_libraries       (multiblock_noblend_continuous          ${libraries})
target_link_libraries       (multiblock_blend_continuous            ${libraries})
target_link_libraries       (multiblock_blend_discrete              ${libraries})

set_property                (TARGET fixed                           PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET multiblock_noblend_continuous   PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET multiblock_blend_continuous     PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property                (TARGET multiblock_blend_discrete       PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS fixed multiblock_noblend_continuous multiblock_blend_continuous multiblock_blend_discrete
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/fixed/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
