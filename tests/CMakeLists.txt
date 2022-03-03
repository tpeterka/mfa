include_directories         (${CMAKE_SOURCE_DIR}/examples)          # needed for block.hpp and opts.h
include_directories         (${CMAKE_SOURCE_DIR}/examples/convert)  # needed for writer.hpp

add_executable              (fixed-test                             fixed.cpp)
target_link_libraries       (fixed-test                             ${libraries})

add_executable              (decode-grid-test                       decode_grid.cpp)
target_link_libraries       (decode-grid-test                       ${libraries})

add_executable              (fixed-multiblock-test                  fixed_multiblock.cpp)
target_link_libraries       (fixed-multiblock-test                  ${libraries})

add_executable              (multi-bc-test                          multi_bc.cpp)
target_link_libraries       (multi-bc-test                          ${libraries})

add_executable              (fixed-s3d-blend                        s3d_blend_test.cpp)
target_link_libraries       (fixed-s3d-blend                        ${libraries})

add_executable              (adaptive-test                          adaptive.cpp)
target_link_libraries       (adaptive-test                          ${libraries})

add_executable              (differentiate-test                     differentiate.cpp)
target_link_libraries       (differentiate-test                     ${libraries})

add_executable              (write-vtk-test                         write_vtk.cpp ${CMAKE_SOURCE_DIR}/examples/convert/writer.cpp)
target_link_libraries       (write-vtk-test                         ${libraries})

add_executable              (vol-iterator-test                      vol_iterator.cpp)
target_link_libraries       (vol-iterator-test                      ${libraries})

add_test                    (NAME fixed-sinc-2d-test
                             COMMAND fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0
                            )
                            
add_test                    (NAME multi-bc-test1
                             COMMAND multi-bc-test -i sinc -d 3 -m 2 -b 2 -o 0.11
                            )

add_test                    (NAME fixed-sinc-2d-test-grid
                             COMMAND fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0 -u 20 -u 20
                            )

add_test                    (NAME fixed-sinc-2d-unstructured-test
                             COMMAND fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0 -x 0
                            )

add_test                    (NAME fixed-sinc-2d-random-test
                             COMMAND fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0 -x 0 -y 4444
                            )

add_test                    (NAME fixed-s3d-1d-test
                             COMMAND fixed-test -i s3d -d 2 -m 1 -p 1 -q 3 -w 0 -f ${s3d_infile}
                            )

add_test                    (NAME fixed-s3d-2d-test
                             COMMAND fixed-test -i s3d -d 3 -m 2 -p 1 -q 3 -w 0 -f ${s3d_infile}
                            )

add_test                    (NAME fixed-s3d-2d-unstructured-test
                             COMMAND fixed-test -i s3d -d 3 -m 2 -p 1 -q 3 -w 0 -f ${s3d_infile} -x 0
                            )

add_test                    (NAME decode-grid1
                             COMMAND decode-grid-test -a 10 -b 0 -c 0 -r 10 -s 30 -t 25 -x 1 -y 1 -z 1 -v 6 -W -f ${s3d_infile} -u 20 -u 20 -u 20
                            )

add_test                    (NAME differentiate-fixed-test
                             COMMAND differentiate-test -i approx.mfa -d 1
                            )

add_test                    (NAME write-vtk-fixed-test
                             COMMAND write-vtk-test approx.mfa
                            )

add_test                    (NAME vol-iterator-test
                             COMMAND vol-iterator-test
                            )

foreach                     (p 1 3 4)
    foreach                 (b 4 16)
        add_test            (NAME fixed-multiblock-test-strong-no-exchange-p${p}-b${b}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 500 -v 50 -b ${b} -t 1 -o 0.0 -w 0
                            )
        add_test            (NAME fixed-multiblock-test-strong-exchange-p${p}-b${b}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 500 -v 50 -b ${b} -t 1 -o 0.1 -w 0
                            )
        add_test            (NAME fixed-multiblock-test-weak-no-exchange-p${p}-b${b}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 100 -v 10 -b ${b} -t 0 -o 0.0 -w 0
                            )
        add_test            (NAME fixed-multiblock-test-weak-exchange-p${p}-b${b}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 100 -v 10 -b ${b} -t 0 -o 0.1 -w 0
                            )
    endforeach              (b)
endforeach                  (p)

add_test                    (NAME write-vtk-multiblock-fixed-test
                             COMMAND write-vtk-test approx.mfa
                            )

foreach                     (p 1 2 3 6)
    add_test                (NAME s3d-blend-p${p}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-s3d-blend> -a 10 -b 0 -c 0 -r 10 -s 100 -t 60 -x 1 -y 2 -z 3 -v 6 -u 30 -u 30 -u 30 -f ${s3d_infile}
                            )
endforeach                  (p)

foreach                     (p 1 2 3 6)
    add_test                (NAME s3d-blend-t-p${p}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-s3d-blend> -a 10 -b 0 -c 0 -r 10 -s 100 -t 60 -x 1 -y 2 -z 3 -v 6 -u 30 -u 30 -u 30 -T 1 -f ${s3d_infile}
                            )
endforeach                  (p)


foreach                     (p 1 2 3 6)
    add_test                (NAME s3d-blend-3d-p${p}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-s3d-blend> -a 10 -b 0 -c 0 -r 40 -s 50 -t 60 -x 1 -y 2 -z 3 -v 6 -u 30 -u 30 -u 30 -T 0 -f ${s3d_infile}
                            )
endforeach                  (p)

foreach                     (p 1 2 3 6)
    add_test                (NAME s3d-blend-3d-t-p${p}
                            COMMAND ${MPIEXEC} -np ${p} $<TARGET_FILE:fixed-s3d-blend> -a 10 -b 0 -c 0 -r 40 -s 50 -t 60 -x 1 -y 2 -z 3 -v 6 -u 30 -u 30 -u 30 -T 1 -f ${s3d_infile}
                            )
endforeach                  (p)

add_test                    (NAME adaptive-sinc-test
                             COMMAND adaptive-test -i sinc -d 3 -m 2 -p 1 -q 5 -e 1e-2 -w 0
                            )

add_test                    (NAME differentiate-adaptive-test
                             COMMAND differentiate-test -i approx.mfa -d 1
                            )

add_test                    (NAME write-vtk-adaptive-test
                             COMMAND write-vtk-test approx.mfa
                            )

add_test                    (NAME adaptive-s3dtest
                             COMMAND adaptive-test -i s3d -d 2 -m 1 -p 1 -q 3 -e 1e-2 -w 0 -f ${s3d_infile}
                            )


