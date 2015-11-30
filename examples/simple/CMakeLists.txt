add_executable              (1d      1d.cpp          )
target_link_libraries       (1d      mfa ${libraries})

install(TARGETS 1d
        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/simple/
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_WRITE GROUP_EXECUTE
        WORLD_READ WORLD_WRITE WORLD_EXECUTE)
