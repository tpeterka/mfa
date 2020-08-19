# Build mfa/_mfa.so
file                        (GLOB MFA_SOURCES "*.cpp")      # lazy
pybind11_add_module         (_mfa ${MFA_SOURCES})
set_target_properties       (_mfa PROPERTIES OUTPUT_NAME _mfa)

