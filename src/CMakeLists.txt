# Buld mfa library

set                     (MFA_SOURCES mfa.cpp encode.cpp decode.cpp new_knots.cpp)

add_library             (mfa ${MFA_SOURCES})
target_link_libraries   (mfa ${libraries})

install                 (TARGETS mfa
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_WRITE GROUP_EXECUTE
                        WORLD_READ WORLD_WRITE WORLD_EXECUTE)
