#include <blaze/Math.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include "TDCR.hpp"

template <size_t N>
bool saveMatrixToCSV(const blaze::StaticMatrix<double, N, 3UL> &mat, const std::string &filepath);

int main()
{
    // constexpr double E = 15.46795366474113845825195E9; // 22.00E9 Young's modulus [GPa] 15.46795366474113845825195E9
    // constexpr double nu = 0.385;                       // Poisson's ratio --> for Nitinol: [0.30, 0.50]
    // constexpr double G = E / (2.00 * (1.00 + nu));     // Shear modulus [GPa]
    // constexpr double radius = 4.001769660928E-4;       // radius of center beam [m] -- 4.6769660928E-4;
    // constexpr double mass = 0.516E-3 * 10;             // total mass of the sheath [kg]
    // constexpr double length = 0.05436;                 // total length of the sheath [m]
    // constexpr double tendonOffset = 1.112394E-3;

    constexpr double E = 15.46795366474113845825195E9; // 22.00E9 Young's modulus [GPa] 15.46795366474113845825195E9
    constexpr double nu = 0.385;                       // Poisson's ratio --> for Nitinol: [0.30, 0.50]
    constexpr double G = E / (2.00 * (1.00 + nu));     // Shear modulus [GPa]
    constexpr double radius = 0.0010;       // radius of center beam [m] -- 4.6769660928E-4;
    constexpr double mass = 0.0040;             // total mass of the sheath [kg]
    constexpr double length = 1.0;                 // total length of the sheath [m]
    constexpr double tendonOffset = 1.112394E-3;

    static constexpr size_t numTendons = 4UL;
    static constexpr size_t backbonePoints = 200UL;

    blaze::StaticMatrix<double, backbonePoints, 3UL> P;

    TDCR<backbonePoints, numTendons> robot = TDCR<backbonePoints, numTendons>(E, G, radius, mass, length, tendonOffset);

    blaze::StaticVector<double, 13UL> baseState;

    blaze::StaticVector<double, 3UL> fe = {0.0, 0.0, 0.0};
    blaze::StaticVector<double, numTendons> tau = {0.0, 10.0, 5.0, 0.0};

    // robot.update_point_force(fe);
    robot.update_initial_guess(tau);
    robot.setTendonPull(tau);
    // robot.test();
    robot.solveBVP();
    robot.getBackbone(P);

    robot.getBaseState(baseState);

    saveMatrixToCSV<backbonePoints>(P, "../examples/output/backbone.csv");
}

template <size_t N>
bool saveMatrixToCSV(const blaze::StaticMatrix<double, N, 3UL> &mat, const std::string &filepath)
{
    std::filesystem::path pathObj(filepath);
    auto parent_path = pathObj.parent_path();

    // Create directory if it does not exist
    if (!parent_path.empty() && !std::filesystem::exists(parent_path))
    {
        if (!std::filesystem::create_directories(parent_path))
        {
            std::cerr << "Failed to create directory: " << parent_path << std::endl;
            return false;
        }
    }

    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }

    file << "x,y,z\n"; // CSV header
    file << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < mat.rows(); ++i)
    {
        for (size_t j = 0; j < mat.columns(); ++j)
        {
            file << mat(i, j);
            if (j < mat.columns() - 1)
                file << ",";
        }
        file << "\n";
    }
    file.close();
    return true;
}