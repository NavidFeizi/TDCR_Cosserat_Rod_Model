#pragma once

#include <iostream>
#include <blaze/Math.h>
#include <cmath>
#include <functional>
#include <chrono>

#include "MathOperations.hpp"
#include "ODEsystems.hpp"

enum class RootFindingMethod
{
    NEWTON_RAPHSON,
    LEVENBERG_MARQUARDT,
    GRAD_FREE,
};

template <size_t N>
class TDCR
{
public:
    TDCR(double E, double G, double radius, double mass, double length);

    void update_point_force(const blaze::StaticVector<double, 3UL> &f);

    /// @brief Solve BVP using shooting method
    void solveBVP();

    void getBackbone(blaze::StaticMatrix<double, N, 3UL> &P);

    void getTipPos(blaze::StaticVector<double, 3UL> &tipPos);

private:
    void residuFunc(const blaze::StaticVector<double, 6UL> &initGuess, blaze::StaticVector<double, 6UL> &residual);

    void odeFunction(const blaze::StaticVector<double, 13UL> &y, blaze::StaticVector<double, 13UL> &dyds) const;

    void computeJacobian(const blaze::StaticVector<double, 6UL> &initGuess, const double eps, blaze::StaticMatrix<double, 6UL, 6UL> &jac);

    blaze::StaticMatrix<double, 13UL, N> m_Y;
    blaze::StaticVector<double, 3UL> s_vStar = {0, 0, 1};
    blaze::StaticVector<double, 3UL> m_u_star = {0, 0, 0};
    blaze::StaticMatrix<double, 3UL, 3UL> m_Kse;
    blaze::StaticMatrix<double, 3UL, 3UL> m_Kbt;
    blaze::StaticVector<double, 3UL> m_nL;
    blaze::StaticVector<double, 3UL> m_mL;
    blaze::StaticVector<double, 3UL> m_f = blaze::StaticVector<double, 3UL>(0.0);                         // external force
    blaze::StaticVector<double, 3UL> m_l = blaze::StaticVector<double, 3UL>(0.0);                         // external moment
    blaze::StaticVector<double, 3UL> s_v1 = blaze::StaticVector<double, 3UL>(0.0);                         // Distal end force
    blaze::StaticVector<double, 3UL> s_u1 = blaze::StaticVector<double, 3UL>(0.0);                         // Distal end torque
    const blaze::StaticVector<double, 3UL> gravity = {0.0, -9.8, 0.0}; // external  load at the tip
    double m_length, m_ds;
    RootFindingMethod m_bvpSolverMthod;
    std::unique_ptr<OdeSystems<13UL, N>> m_ode;
};

template <size_t N>
TDCR<N>::TDCR(double E, double G, double radius, double mass, double length)
{
    double Ixx = M_PI_4 * pow(radius, 4);
    double Izz = Ixx * 2;
    double Area = M_PI * pow(radius, 2);
    // bending and torsion stiffness matrix
    m_Kbt = blaze::StaticMatrix<double, 3UL, 3UL>(0.0);
    m_Kbt(0, 0) = m_Kbt(1, 1) = E * Ixx;
    m_Kbt(2, 2) = G * Izz;
    // shear and extension stiffness matrix
    m_Kse = blaze::StaticMatrix<double, 3UL, 3UL>(0.0);
    m_Kse(0, 0) = G * Area;
    m_Kse(1, 1) = G * Area;
    m_Kse(2, 2) = E * Area;

    m_length = length;
    m_ds = m_length / N;

    m_ode = std::make_unique<OdeSystems<13UL, N>>(IntegrationMethod::RK4);

    m_bvpSolverMthod = RootFindingMethod::NEWTON_RAPHSON;

    m_f = mass / length * gravity;
    m_l = {0.0, 0.0, 0.0};
}

template <size_t N>
void TDCR<N>::update_point_force(const blaze::StaticVector<double, 3UL> &fe1)
{
    s_v1 = fe1;
    s_u1 = {0.0, 0.0, 0.0};
    
}

template <size_t N>
void TDCR<N>::solveBVP()
{
    // 1. Set base BCs (e.g., p(0), R(0))
    // 2. Set desired tip BCs (usually free, so zero force/moment at tip)
    // 3. Integrate to tip
    // 4. Evaluate residual: tip state minus desired tip state (force/moment)
    // 5. Adjust guess (e.g., with finite differences, or numerical solver)
    // Loop until residual is small

    double tol = 1e-8;
    int maxIter = 100;
    double eps = 1e-9;

    blaze::StaticVector<double, 6UL> initGuess(0.0);
    blaze::StaticVector<double, 6UL> residual;
    blaze::StaticMatrix<double, 6UL, 6UL> jac;
    blaze::StaticVector<double, 6UL> delta;

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < maxIter; iter++)
    {
        residuFunc(initGuess, residual);
        double err = blaze::length(residual);
        // std::cout << "Iter " << iter << ", residual: " << err << std::endl;

        if (err < tol)
        {
            blaze::StaticVector<double, 13UL> yl = column(m_Y, N - 1);
            std::cout << "Converged!" << std::endl;
            std::cout << "Iter " << iter << ", residual: " << err << std::endl;
            std::cout << "Tip: " << blaze::trans(subvector(yl, 0, 3)) << std::endl;
            break;
        }

        switch (m_bvpSolverMthod)
        {
        case RootFindingMethod::NEWTON_RAPHSON:
            computeJacobian(initGuess, eps, jac);
            delta = blaze::inv(jac) * residual; // J * delta = residual
            initGuess -= delta;
            break;

        case RootFindingMethod::GRAD_FREE:
            // Update base guess (very basic gradient-free approach)
            subvector(initGuess, 0, 3) -= 0.5 * subvector(residual, 0, 3);
            subvector(initGuess, 3, 3) -= 0.5 * subvector(residual, 3, 3);
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    std::cout << "BVP solve time: " << duration.count() << " [us]" << std::endl;
}

template <size_t N>
void TDCR<N>::computeJacobian(const blaze::StaticVector<double, 6UL> &initGuess, const double eps, blaze::StaticMatrix<double, 6UL, 6UL> &jac)
{
    // I have tested OpenMP but the overhead dor 6X6 jacobian is too much with respect to the serial loop
    blaze::StaticVector<double, 6UL> residual;
    blaze::StaticMatrix<double, 6UL, 6UL> residualPlus, residualMinus;
    blaze::StaticVector<double, 6UL> residual_i;
    blaze::StaticVector<double, 6UL> perturbedGuess;

    auto start = std::chrono::high_resolution_clock::now();
    // Forward difference
    for (size_t i = 0; i < initGuess.size(); i++)
    {
        perturbedGuess = initGuess;
        perturbedGuess[i] += eps;

        this->residuFunc(perturbedGuess, residual_i);
        column(residualPlus, i) = residual_i;
    }

    // Backward difference
    for (size_t i = 0; i < initGuess.size(); i++)
    {
        perturbedGuess = initGuess;
        perturbedGuess[i] -= eps;
        this->residuFunc(perturbedGuess, residual_i);
        column(residualMinus, i) = residual_i;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    // std::cout << "Jacobian time: " << duration.count() << " [us]" << std::endl;
    jac = 0.5 * (residualPlus - residualMinus) / eps;
}

template <size_t N>
void TDCR<N>::residuFunc(const blaze::StaticVector<double, 6UL> &initGuess, blaze::StaticVector<double, 6UL> &residual)
{

    // 1. Set base BCs (e.g., p(0), R(0))
    blaze::StaticVector<double, 13UL> y0;
    blaze::StaticVector<double, 3UL> p0(0.0);                   // Base position
    blaze::StaticVector<double, 4UL> h0 = {1.0, 0.0, 0.0, 0.0}; // Base orientation
    subvector(y0, 0, 3) = p0;
    subvector(y0, 3, 4) = h0;
    subvector(y0, 7, 6) = initGuess; // Initial guess for base force & moment

    // 2. Set desired tip BCs (usually free, so zero force/moment at tip)
    // blaze::StaticVector<double, 3UL> n1 = {0.0, 0.0, 0.0}; // Distal end force
    // blaze::StaticVector<double, 3UL> m1 = {0.0, 0.0, 0.0}; // Distal end torque

    m_ode->solve([this](const blaze::StaticVector<double, 13UL> &y, blaze::StaticVector<double, 13UL> &dyds)
                 { this->odeFunction(y, dyds); },
                 y0, m_length, m_Y);

    // Evaluate residual at the tip
    blaze::StaticVector<double, 13UL> yl = column(m_Y, N - 1);
    blaze::StaticVector<double, 3UL> n1_est = subvector(yl, 7, 3);
    blaze::StaticVector<double, 3UL> m1_est = subvector(yl, 10, 3);

    subvector(residual, 0, 3) = n1_est - s_v1;
    subvector(residual, 3, 3) = m1_est - s_u1;
}

template <size_t N>
void TDCR<N>::odeFunction(const blaze::StaticVector<double, 13UL> &y, blaze::StaticVector<double, 13UL> &dyds) const
{
    // states order p(3), h(4), n(3), m(3)
    blaze::StaticMatrix<double, 3, 3> R = MathOp::getSO3(subvector(y, 3, 4));
    blaze::StaticVector<double, 4UL> h = subvector(y, 3, 4);
    blaze::StaticVector<double, 3UL> v = blaze::inv(m_Kse) * (blaze::trans(R) * subvector(y, 7, 3)) + s_vStar;
    blaze::StaticVector<double, 3UL> u = blaze::inv(m_Kbt) * (blaze::trans(R) * subvector(y, 10, 3)) + m_u_star;
    subvector(dyds, 0, 3) = R * v;
    // subvector(dyds, 3, 4) = 0.5 * MathOp::quaternionHat(u) * h;
    subvector(dyds, 3, 4) = MathOp::quaternionDiff(u, h);
    subvector(dyds, 7, 3) = -m_f;
    subvector(dyds, 10, 3) = -blaze::cross(subvector(dyds, 0, 3), subvector(y, 7, 3)) - m_l;
}

template <size_t N>
void TDCR<N>::getBackbone(blaze::StaticMatrix<double, N, 3UL> &backbone)
{
    // Get the first 3 rows of m_Y which contains the backbone positions
    backbone = blaze::trans(submatrix(m_Y, 0, 0, 3, N));
}

template <size_t N>
void TDCR<N>::getTipPos(blaze::StaticVector<double, 3UL> &tipPos)
{
    // Get the last column of m_Y which contains the tip position
    tipPos = subvector(column(m_Y, N - 1), 0, 3);
}