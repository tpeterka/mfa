add_executable              (nonlinear      nonlinear.cpp)
target_link_libraries       (nonlinear      ${libraries})

set_property                (TARGET  nonlinear PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install                     (TARGETS nonlinear
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/nonlinear/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
