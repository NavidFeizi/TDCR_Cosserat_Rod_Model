#pragma once

#include <functional>
#include <iostream>
#include <blaze/Math.h>

enum class IntegrationMethod
{
    RK4,
    EULER
};

template <size_t D, size_t N>
class OdeSystems
{
public:
    using State = blaze::StaticVector<double, D>;
    using StateMatrix = blaze::StaticMatrix<double, D, N>;

    // Default constructor
    OdeSystems();

    // Overloaded constructor
    OdeSystems(const IntegrationMethod method);

    // Select integration method
    void setIntegrationMethod(const IntegrationMethod method);

    /**
     * @brief Solves an ordinary differential equation (ODE) system over a specified length.
     * @param ode      The ODE system to solve, represented as a function taking the current state and returning its derivative.
     * @param y0       The initial state vector at the start of integration.
     * @param length   The length of the interval over which to integrate the ODE.
     * @param Y        Output parameter to store the computed state trajectory along the integration interval.
     */
    void solve(const std::function<void(const State &, State &)> &ode, const State &y0, const double length, StateMatrix &Y);

private:
    void rk4_integrate(std::function<void(const double, const State &, State &)> ode, StateMatrix &Y, double s0, double sl);

    void euler_integrate(std::function<void(const double, const State &, State &)> ode, StateMatrix &Y, double s0, double sl);

    IntegrationMethod m_method;
};

template <size_t D, size_t N>
OdeSystems<D, N>::OdeSystems()
{
    setIntegrationMethod(IntegrationMethod::RK4); // default integration method
}

template <size_t D, size_t N>
OdeSystems<D, N>::OdeSystems(const IntegrationMethod method)
{
    setIntegrationMethod(method);
}

template <size_t D, size_t N>
void OdeSystems<D, N>::setIntegrationMethod(const IntegrationMethod method)
{
    m_method = method;
}

template <size_t D, size_t N>
void OdeSystems<D, N>::solve(
    const std::function<void(const State &, State &)> &ode,
    const State &y0,
    const double length,
    StateMatrix &Y)
{
    column(Y, 0) = y0;
    switch (m_method)
    {
    case IntegrationMethod::RK4:
        rk4_integrate(
            [ode](const double s_, const State &y_, State &dyds_)
            {
                ode(y_, dyds_);
            },
            Y,
            0, length);
        break;
    case IntegrationMethod::EULER:
        break;
    }
}

template <size_t D, size_t N>
void OdeSystems<D, N>::rk4_integrate(std::function<void(const double, const State &, State &)> ode, StateMatrix &Y, double s0, double sl)
{
    const double ds = static_cast<double>(sl / (N - 1));
    const double half_ds = ds / 2.00;
    const double sixth_ds = ds / 6.00;

    State k1, k2, k3, k4;
    State y = column(Y, 0);
    double s = s0;

    for (int i = 0; i < N - 1; ++i)
    {
        ode(s, y, k1);
        ode(s + half_ds, y + half_ds * k1, k2);
        ode(s + half_ds, y + half_ds * k2, k3);
        ode(s + ds, y + ds * k3, k4);
        y += sixth_ds * (k1 + 2 * k2 + 2 * k3 + k4);
        column(Y, i + 1) = y;
        s += ds;
    }
}

template <size_t D, size_t N>
void OdeSystems<D, N>::euler_integrate(std::function<void(const double, const State &, State &)> ode, StateMatrix &Y, double s0, double sl)
{
    // DO To: to be implemented
    std::cout << "Euler intergration to be implemented..." << std::endl;
}