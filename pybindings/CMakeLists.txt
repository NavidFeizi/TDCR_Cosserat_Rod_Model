# bindings/CMakeLists.txt


find_package(pybind11 REQUIRED)
if(pybind11_FOUND)
    message(STATUS "pybind11 found")
    # Add Python module (header-only binding logic in pybindings.hpp)
    pybind11_add_module(tdcr_cpp MODULE ${CMAKE_CURRENT_SOURCE_DIR}/pybindings.cpp)
    # target_include_directories(tdcr_cpp PRIVATE TDCRLib)
    target_link_libraries(tdcr_cpp PRIVATE TDCRLib)
else()
    message(FATAL_ERROR "pybind11 not found")
endif()

set_target_properties(tdcr_cpp PROPERTIES PREFIX "" SUFFIX ".so")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(Python3_FOUND)
    # Install the Python module to the site-packages directory of the current Python environment
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python module tdcr_cpp will be installed to: ${PYTHON_SITE_PACKAGES}")
    install(TARGETS tdcr_cpp LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}")
    install(CODE "message(STATUS \"Installed tdcr_cpp module to: ${PYTHON_SITE_PACKAGES}/tdcr_cpp.so\")")
else()
    message(FATAL_ERROR "Python3 not found")
endif()

