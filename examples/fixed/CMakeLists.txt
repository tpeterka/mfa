add_executable              (fixed              fixed.cpp)
add_executable              (fixed_multiblock   fixed_multiblock.cpp)
target_link_libraries       (fixed              mfa ${libraries})
target_link_libraries       (fixed_multiblock   mfa ${libraries})

install                     (TARGETS fixed fixed_multiblock
    DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/fixed/
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE
    WORLD_READ WORLD_WRITE WORLD_EXECUTE)
