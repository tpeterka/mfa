add_executable              (n-d      test.cpp          )
target_link_libraries       (n-d      mfa ${libraries})

install(TARGETS n-d
        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/n-d/
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_WRITE GROUP_EXECUTE
        WORLD_READ WORLD_WRITE WORLD_EXECUTE)
