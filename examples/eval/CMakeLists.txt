include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h

add_executable              (differentiate_block        differentiate_block.cpp)
target_link_libraries       (differentiate_block        ${libraries})

add_executable              (eval                       eval.cpp)
target_link_libraries       (eval                       ${libraries})

add_executable              (integrate                  integrate.cpp)
target_link_libraries       (integrate                  ${libraries})

add_executable              (decode_speed               decode_speed.cpp)
target_link_libraries       (decode_speed               ${libraries})

set_property                (TARGET  differentiate_block eval integrate decode_speed PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install                     (TARGETS differentiate_block eval integrate decode_speed
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/eval/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)
