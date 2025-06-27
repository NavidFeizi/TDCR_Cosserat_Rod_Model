// robot_bindings.hpp
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "TDCR.hpp"

namespace py = pybind11;

template <size_t N, size_t numTendons>
/**
 * @brief Binds the TDCR class and its methods to a Python module using pybind11.
 *
 * This function exposes the TDCR class template and its relevant methods to Python,
 * allowing for interaction with the TDCR Cosserat rod model from Python code.
 *
 * Exposed methods:
 * - Constructor: Initializes a TDCR object with geometric and physical parameters.
 * - update_point_force: Updates the external point force applied to the rod.
 * - update_initial_guess: Updates the initial guess for the tendon tensions.
 * - set_tendon_pull: Sets the tendon pull values.
 * - solve_bvp: Solves the boundary value problem for the rod configuration.
 * - get_backbone: Retrieves the backbone coordinates of the rod as a 2D array.
 * - get_base_state: Returns the state vector at the base of the rod.
 * - get_tip_state: Returns the state vector at the tip of the rod.
 * - get_tip_pos: Returns the position of the rod tip.
 * - ode_function: Computes the ODE function for the rod's state and tendon tensions.
 *
 * @param m The Python module to which the TDCR class and its methods are bound.
 */
void bind_tdcr(py::module_ &m)
{
    using TDCRType = TDCR<N, numTendons>;

    py::class_<TDCRType>(m, "TDCR")
        .def(py::init<double, double, double, double, double, double>())

        .def("update_point_force", [](TDCRType &self, std::vector<double> f)
             {
            blaze::StaticVector<double, 3UL> fe{f[0], f[1], f[2]};
            self.update_point_force(fe); })

        .def("update_initial_guess", [](TDCRType &self, std::vector<double> tau)
             {
            blaze::StaticVector<double, numTendons> tau_vec;
            for (size_t i = 0; i < numTendons; ++i)
                tau_vec[i] = tau[i];
            self.update_initial_guess(tau_vec); })

        .def("set_tendon_pull", [](TDCRType &self, std::vector<double> tau)
             {
            blaze::StaticVector<double, numTendons> tau_vec;
            for (size_t i = 0; i < numTendons; ++i)
                tau_vec[i] = tau[i];
            self.setTendonPull(tau_vec); })

        .def("solve_bvp", &TDCRType::solveBVP)

        .def("get_backbone", [](TDCRType &self)
             {
            blaze::StaticMatrix<double, N, 3UL> P;
            self.getBackbone(P);
            std::vector<std::vector<double>> result(N, std::vector<double>(3));
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < 3; ++j)
                    result[i][j] = P(i, j);
            return result; })

        .def("get_base_state", [](TDCRType &self)
             {
            blaze::StaticVector<double, 13UL> state;
            self.getBaseState(state);
            return std::vector<double>(state.begin(), state.end()); })

        .def("get_tip_state", [](TDCRType &self)
             {
            blaze::StaticVector<double, 13UL> state;
            self.getTipState(state);
            return std::vector<double>(state.begin(), state.end()); })

        .def("get_tip_pos", [](TDCRType &self)
             {
            blaze::StaticVector<double, 3UL> tip;
            self.getTipPos(tip);
            return std::vector<double>(tip.begin(), tip.end()); })

        .def("ode_function", [](TDCRType &self, const std::vector<double> &y, const std::vector<double> &tau)
             {
            blaze::StaticVector<double, 13UL> y_vec;
            for (size_t i = 0; i < 13; ++i)
                y_vec[i] = y[i];
            blaze::StaticVector<double, numTendons> tau_vec;
            for (size_t i = 0; i < numTendons; ++i)
                tau_vec[i] = tau[i];
            blaze::StaticVector<double, 13UL> dyds;
            self.odeFunction(y_vec, tau_vec, dyds);
            return std::vector<double>(dyds.begin(), dyds.end()); });
}
