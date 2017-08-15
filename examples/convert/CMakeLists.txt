add_executable              (write_vtk    write_vtk.cpp)
target_sources              (write_vtk    PUBLIC  writer.cpp)
target_link_libraries       (write_vtk    mfa ${libraries})

# install(TARGETS convert 
#         DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/adaptive/
#         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
#         GROUP_READ GROUP_WRITE GROUP_EXECUTE
#         WORLD_READ WORLD_WRITE WORLD_EXECUTE)
