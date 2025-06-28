#pragma once

#include <iostream>
#include <blaze/Math.h>
#include <cmath>

namespace MathOp
{
    // function that computes the skew-symmetric matrix (hat operator) for SO(3) 
    inline blaze::StaticMatrix<double, 3UL, 3UL> hat(const blaze::StaticVector<double, 3UL> &R3)
    {
        blaze::StaticMatrix<double, 3UL, 3UL> SO3(0.0);
        SO3(0, 1) = -R3[2];
        SO3(0, 2) = R3[1];
        SO3(1, 0) = R3[2];
        SO3(1, 2) = -R3[0];
        SO3(2, 0) = -R3[1];
        SO3(2, 1) = R3[0];
        return SO3;
    }

    // function that computes the skew-symmetric matrix (hat operator) for SE(3)
    // R6 is a 6D vector with the first 3 elements being the translation and the last 3 elements being the rotation
    inline blaze::StaticMatrix<double, 4UL, 4UL> hat(const blaze::StaticVector<double, 6UL> &R6)
    {
        blaze::StaticMatrix<double, 3UL, 3UL> SE3(0.0);
        SE3(0, 1) = -R6[5];
        SE3(0, 2) = R6[4];
        SE3(1, 0) = R6[5];
        SE3(1, 2) = -R6[3];
        SE3(2, 0) = -R6[4];
        SE3(2, 1) = R6[3];
        // v (translation)
        SE3(0, 3) = R6[0];
        SE3(1, 3) = R6[1];
        SE3(2, 3) = R6[2];
        return SE3;
    }

    // function that computes the squared hat operator
    inline blaze::StaticMatrix<double, 3UL, 3UL> hatSqr(const blaze::StaticVector<double, 3UL> &v)
    {
        blaze::StaticMatrix<double, 3UL, 3UL> hatSqr = {{-v[2UL] * v[2UL] - v[1UL] * v[1UL], v[1UL] * v[0UL], v[0UL] * v[2UL]},
                                                        {v[0UL] * v[1UL], -v[2UL] * v[2UL] - v[0UL] * v[0UL], v[2UL] * v[1UL]},
                                                        {v[0UL] * v[2UL], v[1UL] * v[2UL], -v[1UL] * v[1UL] - v[0UL] * v[0UL]}};

        return hatSqr;
    }

    // function that computes the differential quaternion evolution
    inline blaze::StaticVector<double, 4UL> quaternionDiff(const blaze::StaticVector<double, 3UL> &u, const blaze::StaticVector<double, 4UL> &h)
    {
        blaze::StaticVector<double, 4UL> hs;

        hs[0UL] = 0.50 * (-u[0UL] * h[1UL] - u[1UL] * h[2UL] - u[2UL] * h[3UL]);
        hs[1UL] = 0.50 * (u[0UL] * h[0UL] + u[2UL] * h[2UL] - u[1UL] * h[3UL]);
        hs[2UL] = 0.50 * (u[1UL] * h[0UL] - u[2UL] * h[1UL] + u[0UL] * h[3UL]);
        hs[3UL] = 0.50 * (u[2UL] * h[0UL] + u[1UL] * h[1UL] - u[0UL] * h[2UL]);

        return hs;
    }

    // function that computes the quaternion hat operator
    inline blaze::StaticMatrix<double, 4UL, 4UL> quaternionHat(const blaze::StaticVector<double, 3UL> u)
    {
        blaze::StaticMatrix<double, 4UL, 4UL> SO3;
        SO3(0, 0) = 0.0;
        SO3(0, 1) = -u[0];
        SO3(0, 2) = -u[1];
        SO3(0, 3) = -u[2];

        SO3(1, 0) = u[0];
        SO3(1, 1) = 0.0;
        SO3(1, 2) = u[2];
        SO3(1, 3) = -u[1];

        SO3(2, 0) = u[1];
        SO3(2, 1) = -u[2];
        SO3(2, 2) = 0.0;
        SO3(2, 3) = u[0];

        SO3(3, 0) = u[2];
        SO3(3, 1) = u[1];
        SO3(3, 2) = -u[0];
        SO3(3, 3) = 0.0;
        return SO3;
    }

    // function that converts a quaternion to its corresponding SO(3) rotation matrix
    inline blaze::StaticMatrix<double, 3UL, 3UL> getSO3(const blaze::StaticVector<double, 4UL> &h)
    {
        // Extract components
        double h1 = h[0], h2 = h[1], h3 = h[2], h4 = h[3];

        // Compute norm squared
        double h_norm_sq = h1 * h1 + h2 * h2 + h3 * h3 + h4 * h4;

        // Avoid divide by zero
        if (h_norm_sq == 0.0)
            h_norm_sq = 1.0;

        // Compute elements
        blaze::StaticMatrix<double, 3UL, 3UL> R;

        R(0, 0) = 1 - 2.0 * (h3 * h3 + h4 * h4) / h_norm_sq;
        R(0, 1) = 2.0 * (h2 * h3 - h4 * h1) / h_norm_sq;
        R(0, 2) = 2.0 * (h2 * h4 + h3 * h1) / h_norm_sq;

        R(1, 0) = 2.0 * (h2 * h3 + h4 * h1) / h_norm_sq;
        R(1, 1) = 1 - 2.0 * (h2 * h2 + h4 * h4) / h_norm_sq;
        R(1, 2) = 2.0 * (h3 * h4 - h2 * h1) / h_norm_sq;

        R(2, 0) = 2.0 * (h2 * h4 - h3 * h1) / h_norm_sq;
        R(2, 1) = 2.0 * (h3 * h4 + h2 * h1) / h_norm_sq;
        R(2, 2) = 1 - 2.0 * (h2 * h2 + h3 * h3) / h_norm_sq;

        return R;
    }

}