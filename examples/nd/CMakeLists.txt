add_executable              (nd      nd.cpp          )
target_link_libraries       (nd      mfa ${libraries})

install(TARGETS nd
        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/nd/
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_WRITE GROUP_EXECUTE
        WORLD_READ WORLD_WRITE WORLD_EXECUTE)
