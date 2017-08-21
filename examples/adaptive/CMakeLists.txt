add_executable              (adaptive   adaptive.cpp)
target_link_libraries       (adaptive   mfa ${libraries})

install                     (TARGETS adaptive
    DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/adaptive/
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE
    WORLD_READ WORLD_WRITE WORLD_EXECUTE)
