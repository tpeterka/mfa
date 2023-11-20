include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (time_varying                           time_varying.cpp)
target_link_libraries       (time_varying                           ${libraries})
set_property                (TARGET time_varying                    PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install                     (TARGETS time_varying 
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/encode/timeseries
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)