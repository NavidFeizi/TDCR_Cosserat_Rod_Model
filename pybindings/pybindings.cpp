#include "pybindings.hpp"

PYBIND11_MODULE(tdcr_cpp, m)
{
    bind_tdcr<200, 4>(m); // instantiate with your desired parameters
}
