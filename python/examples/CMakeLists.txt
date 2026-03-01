foreach     (fn
                fixed.py
                evaluate.py
                test.py
                practical_ct_pipeline.py
            )

add_custom_target           (${fn} ALL ${CMAKE_COMMAND} -E copy
                                             ${CMAKE_CURRENT_SOURCE_DIR}/${fn} ${CMAKE_CURRENT_BINARY_DIR}/${fn}
                                             DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${fn})

endforeach()
