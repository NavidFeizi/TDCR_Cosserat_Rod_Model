// robot_bindings.hpp
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "TDCR.hpp"

namespace py = pybind11;

template <size_t N, size_t numTendons>
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
