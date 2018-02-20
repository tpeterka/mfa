add_executable              (differentiate      differentiate.cpp)
target_link_libraries       (differentiate      mfa ${libraries})

set_property 				(TARGET  differentiate PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install                     (TARGETS differentiate
    DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/differentiate/
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE
    WORLD_READ WORLD_WRITE WORLD_EXECUTE)
