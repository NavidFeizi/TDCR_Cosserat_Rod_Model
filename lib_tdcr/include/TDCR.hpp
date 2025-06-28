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

template <size_t N, size_t numTendons>
class TDCR
{
public:
    TDCR(double E, double G, double radius, double mass, double length, double tendonOffset);

    /// @brief to be developed
    void update_point_force(const blaze::StaticVector<double, 3UL> &f);

    /// @brief Rough estimate of the initial guess for the BVP solver from the tendon pulling forces.
    void update_initial_guess(const blaze::StaticVector<double, numTendons> &tau);

    /// @brief Solves the boundary value problem (BVP) using the shooting method.
    void solveBVP();

    /// @brief Retrieves the backbone positions of the rod.
    void getBackbone(blaze::StaticMatrix<double, N, 3UL> &P);

    /// @brief Retrieves the state vector at the base of the rod.
    void getBaseState(blaze::StaticVector<double, 13UL> &baseState);

    /// @brief Retrieves the state vector at the tip of the rod.
    void getTipState(blaze::StaticVector<double, 13UL> &tipState);

    /// @brief Retrieves the position of the rod tip.
    void getTipPos(blaze::StaticVector<double, 3UL> &tipPos);

    /// @brief Sets the tendon pulling forces. Must be called before solving the BVP.
    void setTendonPull(blaze::StaticVector<double, numTendons> &tau);

    /**
     * @brief Computes the derivative of the Cosserat rod state vector for use in ODE integration.
     * @param y     The current state vector of the Cosserat rod (size 13) [p, h, v, u].
     * @param tau   The tendon tensions applied to the rod (size numTendons).
     * @param dyds  Output parameter to store the computed derivative of the state vector (size 13).     */
    void odeFunction(const blaze::StaticVector<double, 13UL> &y,
                     const blaze::StaticVector<double, numTendons> &tau,
                     blaze::StaticVector<double, 13UL> &dyds) const;

private:
    ///@brief Integrates the BVP and computes the residual states error as the tip of the rod.
    void residuFunc(const blaze::StaticVector<double, 6UL> &initGuess, blaze::StaticVector<double, 6UL> &residual);

    ///@brief Computes the Jacobian matrix for the BVP solver using center point finite differences.
    void computeJacobian(const blaze::StaticVector<double, 6UL> &initGuess, const double eps, blaze::StaticMatrix<double, 6UL, 6UL> &jac);

    blaze::StaticMatrix<double, 13UL, N> m_Y;                                      // state matrix, rows: [p, h, v, u] | cols: states at a point along the backbone
    blaze::StaticVector<double, 3UL> s_vStar = {0.0, 0.0, 1.0};                    // reference shear and extension strain
    blaze::StaticVector<double, 3UL> s_uStar = {0.0, 0.0, 0.0};                    // reference bending and torsion strain
    blaze::StaticMatrix<double, 3UL, 3UL> m_Kse, m_Kbt;                            // shear and extension, bending and torsion stiffness matrices
    blaze::StaticMatrix<double, 3UL, 3UL> m_1_Kse, m_1_Kbt;                        // inverse of shear and extension, bending and torsion stiffness matrices
    blaze::StaticVector<double, 6UL> m_initGuess = {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}; // initial guess for the base force and moment
    blaze::StaticVector<double, 3UL> m_f = blaze::StaticVector<double, 3UL>(0.0);  // external force
    blaze::StaticVector<double, 3UL> m_l = blaze::StaticVector<double, 3UL>(0.0);  // external moment
    blaze::StaticVector<double, 3UL> s_v1 = blaze::StaticVector<double, 3UL>(0.0); // Distal end force
    blaze::StaticVector<double, 3UL> s_u1 = blaze::StaticVector<double, 3UL>(0.0); // Distal end torque
    const blaze::StaticVector<double, 3UL> gravity = {0.0, -9.8, 0.0};             // external  load at the tip
    double m_length, m_ds;
    double m_tendonOffset;
    RootFindingMethod m_bvpSolverMthod;
    std::unique_ptr<OdeSystems<13UL, N>> m_ode;

    blaze::StaticVector<blaze::StaticVector<double, 3UL>, numTendons> m_r;
    blaze::StaticVector<double, numTendons> m_tau;
};

template <size_t N, size_t numTendons>
TDCR<N, numTendons>::TDCR(double E, double G, double radius, double mass, double length, double tendonOffset)
{
    double Ixx = M_PI_4 * pow(radius, 4);
    double Izz = Ixx * 2;
    double Area = M_PI * pow(radius, 2);
    // bending and torsion stiffness matrix
    m_Kbt = blaze::StaticMatrix<double, 3UL, 3UL>(0.0);
    m_Kbt(0, 0) = m_Kbt(1, 1) = E * Ixx;
    m_Kbt(2, 2) = G * Izz;
    m_1_Kbt = blaze::inv(m_Kbt);

    // shear and extension stiffness matrix
    m_Kse = blaze::StaticMatrix<double, 3UL, 3UL>(0.0);
    m_Kse(0, 0) = m_Kse(1, 1) = G * Area;
    m_Kse(2, 2) = E * Area;
    m_1_Kse = blaze::inv(m_Kse);

    double rho = mass / length / Area;

    m_tendonOffset = tendonOffset;
    m_length = length;
    m_ds = m_length / N;

    m_ode = std::make_unique<OdeSystems<13UL, N>>(IntegrationMethod::RK4);

    m_bvpSolverMthod = RootFindingMethod::NEWTON_RAPHSON;

    blaze::StaticVector<double, numTendons> angles;
    for (size_t idx = 0UL; idx < numTendons; idx++)
    {
        angles[idx] = 2 * M_PI * idx / numTendons;
    }

    for (size_t idx = 0UL; idx < numTendons; ++idx)
    {
        this->m_r[idx][0UL] = m_tendonOffset * cos(angles[idx]);
        this->m_r[idx][1UL] = m_tendonOffset * sin(angles[idx]);
        this->m_r[idx][2UL] = 0.00;
        // eliminates small numbers (trash) from "zeros" entries of the radial offset vector
        this->m_r[idx] = blaze::map(this->m_r[idx], [](double d)
                                    { return (std::abs(d) < 1.00E-5) ? 0.00 : d; });
    }

    m_f = rho * Area * gravity;
    m_l = {0.0, 0.0, 0.0};
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::update_point_force(const blaze::StaticVector<double, 3UL> &fe1)
{
    // m_f = fe1;
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::update_initial_guess(const blaze::StaticVector<double, numTendons> &tau)
{
    blaze::StaticVector<double, 4UL> h0 = {1.0, 0.0, 0.0, 0.0}; // Initial orientation (identity quaternion)
    blaze::StaticMatrix<double, 3, 3> R0 = MathOp::getSO3(h0);

    blaze::StaticVector<double, 3UL> n0, m0 = blaze::StaticVector<double, 3UL>(0.0);
    for (size_t idx = 0UL; idx < numTendons; ++idx)
    {
        blaze::StaticVector<double, 3UL> fb_i = -tau[idx] * blaze::normalize(s_vStar);
        n0 += fb_i;
        m0 += blaze::cross(m_r[idx], fb_i);
    }
    blaze::StaticVector<double, 3UL> v = m_1_Kse * (blaze::inv(R0) * n0) + s_vStar;
    blaze::StaticVector<double, 3UL> u = m_1_Kbt * (blaze::inv(R0) * m0) + s_uStar;
    m_initGuess = {v[0], v[1], v[2], u[0], u[1], u[2]}; // Initial guess for base force and moment
    // std::cout << "Initial guess: " << blaze::trans(m_initGuess) << std::endl;
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::solveBVP()
{
    // 1. Set base BCs (e.g., p(0), R(0))
    // 2. Set desired tip BCs (usually free, so zero force/moment at tip)
    // 3. Integrate to tip
    // 4. Evaluate residual: tip state minus desired tip state (force/moment)
    // 5. Adjust guess (e.g., with finite differences, or numerical solver)
    // Loop until residual is small

    double tol = 1e-8;
    int maxIter = 1000;
    double eps = 1e-9;

    // blaze::StaticVector<double, 6UL> m_initGuess = {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}; // [v', u']
    blaze::StaticVector<double, 6UL> residual;
    blaze::StaticMatrix<double, 6UL, 6UL> jac;
    blaze::StaticVector<double, 6UL> delta;

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < maxIter; iter++)
    {
        this->residuFunc(m_initGuess, residual);
        double err = blaze::length(residual);

        if (err < tol)
        {
            blaze::StaticVector<double, 13UL> yl = column(m_Y, N - 1);
            // std::cout << "Converged!" << std::endl;
            // std::cout << "Iter " << iter << ", residual: " << err << std::endl;
            // std::cout << "Tip: " << blaze::trans(subvector(yl, 0, 3)) << std::endl;
            break;
        }

        switch (m_bvpSolverMthod)
        {
        case RootFindingMethod::NEWTON_RAPHSON:
            computeJacobian(m_initGuess, eps, jac);
            delta = blaze::inv(jac) * residual; // J * delta = residual
            m_initGuess -= delta;
            break;

        case RootFindingMethod::GRAD_FREE:
            // Update base guess (very basic gradient-free approach)
            subvector(m_initGuess, 0, 3) -= 0.5 * subvector(residual, 0, 3);
            subvector(m_initGuess, 3, 3) -= 0.5 * subvector(residual, 3, 3);
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    // std::cout << "BVP solve time: " << duration.count() << " [us]" << std::endl;
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::computeJacobian(const blaze::StaticVector<double, 6UL> &initGuess, const double eps, blaze::StaticMatrix<double, 6UL, 6UL> &jac)
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

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::residuFunc(const blaze::StaticVector<double, 6UL> &initGuess, blaze::StaticVector<double, 6UL> &residual)
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

    m_ode->solve([this](const blaze::StaticVector<double, 13UL> &y,
                        blaze::StaticVector<double, 13UL> &dyds)
                 { this->odeFunction(y, this->m_tau, dyds); },
                 y0, m_length, m_Y);

    // Evaluate residual at the tip
    blaze::StaticVector<double, 13UL> yl = column(m_Y, N - 1);
    blaze::StaticVector<double, 3UL> v1 = subvector(yl, 7, 3);    // shear extension strain
    blaze::StaticVector<double, 3UL> u1 = subvector(yl, 10, 3);   // bending torsion strain
    blaze::StaticVector<double, 3UL> nb = m_Kse * (v1 - s_vStar); // backbone internal force in body frame
    blaze::StaticVector<double, 3UL> mb = m_Kbt * (u1 - s_uStar); // backbone internal moment in body frame
    blaze::StaticVector<double, 3UL> forceError(-nb);
    blaze::StaticVector<double, 3UL> momentError(-mb);
    blaze::StaticVector<double, 3UL> pbi_s, fb_i;
    for (size_t idx = 0UL; idx < numTendons; ++idx)
    {
        pbi_s = MathOp::hat(u1) * m_r[idx] + v1;
        fb_i = -m_tau[idx] * blaze::normalize(pbi_s);
        forceError += fb_i;
        momentError += blaze::cross(m_r[idx], fb_i);
    }
    subvector(residual, 0, 3) = forceError;
    subvector(residual, 3, 3) = momentError;
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::odeFunction(const blaze::StaticVector<double, 13UL> &y, const blaze::StaticVector<double, numTendons> &tau, blaze::StaticVector<double, 13UL> &dyds) const
{
    // states y = p(3), h(4), v(3), u(3)
    // dyds = [ dp/ds, dh/ds, dv/ds, du/ds ]
    blaze::StaticVector<double, 3UL> p = subvector(y, 0, 3);
    blaze::StaticVector<double, 4UL> h = subvector(y, 3, 4);
    blaze::StaticVector<double, 3UL> v = subvector(y, 7, 3);
    blaze::StaticVector<double, 3UL> u = subvector(y, 10, 3);
    blaze::StaticMatrix<double, 3, 3> R = MathOp::getSO3(h);
    blaze::StaticMatrix<double, 3, 3> uh = MathOp::hat(u);

    decltype(p) dpds;
    decltype(h) dhds;
    // decltype(v) dvds;
    // decltype(u) duds;

    double pbi_sNorm;
    blaze::StaticVector<double, 3UL> pbi_s, ai, a, bi, b, c, d = blaze::StaticVector<double, 3UL>(0.0);
    blaze::StaticMatrix<double, 3UL, 3UL> A, B, Ai, Bi, G, H = blaze::StaticMatrix<double, 3UL, 3UL>(0.0);
    blaze::StaticMatrix<double, 6UL, 6UL> phi(0.0);
    blaze::StaticVector<double, 6UL> rhs(0.0);
    blaze::StaticVector<double, 6UL> lhs(0.0);

    for (size_t idx = 0; idx < numTendons; idx++)
    {
        pbi_s = uh * m_r[idx] + v;
        pbi_sNorm = blaze::norm(pbi_s);
        Ai = -(tau[idx] / pow(pbi_sNorm, 3)) * MathOp::hatSqr(pbi_s);
        Bi = MathOp::hat(m_r[idx]) * Ai;
        A += Ai;
        B += Bi;
        G -= Ai * MathOp::hat(m_r[idx]);
        H -= Bi * MathOp::hat(m_r[idx]);
        ai = Ai * uh * (pbi_s);
        bi = MathOp::hat(m_r[idx]) * ai;
        a += ai;
        b += bi;
    }

    blaze::StaticVector<double, 3UL> nb = m_Kse * (v - s_vStar); // internal force in body frame
    blaze::StaticVector<double, 3UL> mb = m_Kbt * (u - s_uStar); // internal moment in body frame

    subvector(rhs, 0, 3) = 0.0 - uh * nb - blaze::trans(R) * m_f - a;
    subvector(rhs, 3, 3) = 0.0 - uh * mb - MathOp::hat(v) * nb - blaze::trans(R) * m_l - b;

    // assembling the phi matrix
    submatrix(phi, 0UL, 0UL, 3UL, 3UL) = m_Kse + A;
    submatrix(phi, 0UL, 3UL, 3UL, 3UL) = G;
    submatrix(phi, 3UL, 0UL, 3UL, 3UL) = B;
    submatrix(phi, 3UL, 3UL, 3UL, 3UL) = m_Kbt + H;

    dpds = R * v;
    dhds = MathOp::quaternionDiff(u, h);
    // dvds = 0.0 - m_1_Kse * (uh * m_Kse * (v - s_vStar) + blaze::trans(R) * m_f);
    // duds = 0.0 - m_1_Kbt * (uh * m_Kbt * (u - s_uStar) + MathOp::hat(v) * m_Kse * (v - s_vStar) + blaze::trans(R) * m_l);

    lhs = blaze::inv(phi) * rhs;

    subvector(dyds, 0, 3) = dpds;
    subvector(dyds, 3, 4) = dhds;
    // subvector(dyds, 7, 3) = dvds;
    // subvector(dyds, 10, 3) = duds;
    subvector(dyds, 7, 6) = lhs;
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::getBackbone(blaze::StaticMatrix<double, N, 3UL> &backbone)
{
    // Get the first 3 rows of m_Y which contains the backbone positions
    backbone = blaze::trans(submatrix(m_Y, 0, 0, 3, N));
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::getTipPos(blaze::StaticVector<double, 3UL> &tipPos)
{
    // Get the last column of m_Y which contains the tip position
    tipPos = subvector(column(m_Y, N - 1), 0UL, 3UL);
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::getBaseState(blaze::StaticVector<double, 13UL> &baseState)
{
    baseState = subvector(column(m_Y, 0UL), 0UL, 13UL);
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::getTipState(blaze::StaticVector<double, 13UL> &tipState)
{
    // Get the last column of m_Y which contains the tip position
    tipState = subvector(column(m_Y, N - 1), 0UL, 13UL);
}

template <size_t N, size_t numTendons>
void TDCR<N, numTendons>::setTendonPull(blaze::StaticVector<double, numTendons> &tau)
{
    m_tau = tau;
}
