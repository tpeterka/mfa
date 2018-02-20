add_executable              (write_vtk              write_vtk.cpp writer.cpp)
add_executable              (write_vtk_multiblock   write_vtk_multiblock.cpp writer.cpp)
target_link_libraries       (write_vtk              mfa ${libraries})
target_link_libraries       (write_vtk_multiblock   mfa ${libraries})

set_property 				(TARGET write_vtk 				PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property 				(TARGET write_vtk_multiblock 	PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install                     (TARGETS write_vtk write_vtk_multiblock
    DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/convert/
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE
    WORLD_READ WORLD_WRITE WORLD_EXECUTE)
