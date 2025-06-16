#include "pybindings.hpp"

PYBIND11_MODULE(tdcr_physics, m)
{
    bind_tdcr<200, 4>(m); // instantiate with your desired parameters
}
