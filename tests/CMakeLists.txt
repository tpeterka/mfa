include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h
include_directories         (${CMAKE_SOURCE_DIR}/examples/convert)  # needed for writer.hpp

add_executable              (fixed-test                             fixed.cpp)
target_link_libraries       (fixed-test                             ${libraries})

add_executable              (fixed-multiblock-test                  fixed_multiblock.cpp)
target_link_libraries       (fixed-multiblock-test                  ${libraries})

add_executable              (adaptive-test                          adaptive.cpp)
target_link_libraries       (adaptive-test                          ${libraries})

add_executable              (differentiate-test                     differentiate.cpp)
target_link_libraries       (differentiate-test                     ${libraries})

add_executable              (write-vtk-test                         write_vtk.cpp ${CMAKE_SOURCE_DIR}/examples/convert/writer.cpp)
target_link_libraries       (write-vtk-test                         ${libraries})

add_executable              (write-vtk-multiblock-test              write_vtk_multiblock.cpp ${CMAKE_SOURCE_DIR}/examples/convert/writer.cpp)
target_link_libraries       (write-vtk-multiblock-test              ${libraries})

add_test                    (NAME fixed-test
                             COMMAND fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0
                            )

add_test                    (NAME differentiate-fixed-test
                             COMMAND differentiate-test -i approx.out -d 1
                            )

add_test                    (NAME write-vtk-fixed-test
                             COMMAND write-vtk-test approx.out
                            )

foreach                     (p 1 3 4)
    foreach                 (b 4 16)
        add_test            (NAME fixed-multiblock-test-strong-p${p}-b${b}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 500 -v 50 -b ${b} -t 1 -w 0
                            )
        add_test            (NAME fixed-multiblock-test-weak-p${p}-b${b}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 100 -v 10 -b ${b} -t 0 -w 0
                            )
    endforeach              (b)
endforeach                  (p)

add_test                    (NAME write-vtk-multiblock-fixed-test
                             COMMAND write-vtk-multiblock-test approx.out
                            )

add_test                    (NAME adaptive-sinc-test
                             COMMAND adaptive-test -i sinc -d 3 -m 2 -p 1 -q 5 -e 1e-2 -w 0
                            )

add_test                    (NAME differentiate-adaptive-test
                             COMMAND differentiate-test -i approx.out -d 1
                            )

add_test                    (NAME write-vtk-adaptive-test
                             COMMAND write-vtk-test approx.out
                            )

add_test                    (NAME adaptive-s3dtest
                             COMMAND adaptive-test -i s3d -d 2 -m 1 -p 1 -q 3 -e 1e-2 -w 0
                            )


