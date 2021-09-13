#include <gtest/gtest.h>

#include "../BasicLinearAlgebra.h"

using namespace BLA;

TEST(LinearAlgebra, LUDecomposition)
{
    Matrix<7, 7> A = {16, 78, 50, 84, 70, 63, 2, 32, 33, 61, 40, 17, 96, 98, 50, 80, 78, 27, 86, 49, 57, 10, 42, 96, 44,
                      87, 60, 67, 16, 59, 53, 8, 64, 97, 41, 90, 56, 22, 48, 32, 12, 4,  45, 78, 43, 11, 7,  8,  12};

    auto A_orig = A;

    auto decomp = LUDecompose(A);

    EXPECT_FALSE(decomp.singular);

    auto A_reconstructed = decomp.P() * decomp.L() * decomp.U();

    for (int i = 0; i < A.Rows; ++i)
    {
        for (int j = 0; j < A.Cols; ++j)
        {
            EXPECT_FLOAT_EQ(A_reconstructed(i, j), A_orig(i, j));
        }
    }
}

TEST(LinearAlgebra, LUSolution)
{
    Matrix<3, 3> A{2, 5, 8, 0, 8, 6, 6, 7, 5};
    Matrix<3, 1> b{10, 11, 12};
    Matrix<3, 1> x_expected = {0.41826923, 0.97115385, 0.53846154};

    auto decomp = LUDecompose(A);

    auto x = LUSolve(decomp, b);

    for (int i = 0; i < x_expected.Rows; ++i)
    {
        EXPECT_FLOAT_EQ(x_expected(i), x(i));
    }
}

TEST(LinearAlgebra, CholeskyDecomposition)
{
    // clang-format off

    // We could fill in this lower triangle but since A is required to be symmetric they can be inferred
    // from the upper triangle
    Matrix<4, 4> A = {0.60171582, -0.20854924,  0.52925771,  0.24206045,
                      0.0,         0.33012847, -0.28941531, -0.33854164,
                      0.0,         0.0,         3.54506632,  1.56758518,
                      0.0,         0.0,         0.0,         1.75291733};
    // clang-format on

    auto chol = decompose(A);

    // Build a lower triangular matrix and its transpose from the results of the decomposition
    Matrix<4, 4> L, L_T;

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (i > j)
            {
                L(i, j) = L_T(j, i) = chol.A(i, j);
            }
            else if (i == j)
            {
                L(i, i) = L_T(i, i) = chol.diagonal[i];
            }
            else
            {
                L(i, j) = L_T(j, i) = 0;
            }
        }
    }

    EXPECT_TRUE(chol.positive_definite);

    auto A_reconstructed = L * L_T;

    // Compare the recontruction to the upper triangle of A (the lower triangle will be overwritten by decompose)
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (i <= j)
            {
                EXPECT_FLOAT_EQ(A_reconstructed(i, j), A(i, j));
            }
        }
    }
}

TEST(LinearAlgebra, Solve)
{
    Matrix<5, 5> A = {0.78183123,  0.08385324,  0.37172332,  -0.72518705, -1.11317593, 0.08385324, 0.56011595,
                      0.19965695,  -0.17488402, -0.12703805, 0.37172332,  0.19965695,  0.52769031, -0.19284881,
                      -0.45321194, -0.72518705, -0.17488402, -0.19284881, 2.19127456,  2.13045896, -1.11317593,
                      -0.12703805, -0.45321194, 2.13045896,  3.50184434};

    Matrix<5, 5> A_copy = A;

    Matrix<5> b = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto chol = decompose(A);

    auto x = solve(chol, b);

    auto b_expected = A_copy * x;

    for (int i = 0; i < 5; ++i)
    {
        EXPECT_FLOAT_EQ(b_expected(i), b(i));
    }
}

TEST(LinearAlgebra, Inversion)
{
    BLA::Matrix<3, 3> A = {9.79, 9.33, 11.62, 7.77, 14.77, 14.12, 11.33, 15.72, 12.12};

    auto A_inv = A;
    Invert(A_inv);

    auto I = A_inv * A;

    for (int i = 0; i < A.Rows; ++i)
    {
        for (int j = 0; j < A.Cols; ++j)
        {
            if (i == j)
            {
                EXPECT_NEAR(I(i, j), 1.0, 1e-5);
            }
            else
            {
                EXPECT_NEAR(I(i, j), 0.0, 1e-5);
            }
        }
    }
}

TEST(Arithmetic, Determinant)
{
    Matrix<6, 6> B = {0.05508292, 0.82393504, 0.34938018, 0.63818054, 0.18291131, 0.1986636,  0.56799604, 0.81077491,
                      0.71472733, 0.68527613, 0.72759853, 0.25983183, 0.99035713, 0.76096889, 0.26130098, 0.16855372,
                      0.0253581,  0.47907605, 0.58735833, 0.0913456,  0.03221577, 0.5210331,  0.61583369, 0.33233299,
                      0.20578816, 0.356537,   0.70661899, 0.6569476,  0.90074756, 0.59771572, 0.20054716, 0.41290408,
                      0.70679818, 0.321249,   0.81886099, 0.77819212};

    float det_numpy = -0.03919640039505248;

    EXPECT_FLOAT_EQ(Determinant(B), det_numpy);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
