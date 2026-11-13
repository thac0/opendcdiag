/**
 * @file
 *
 * @copyright
 * Copyright 2022 Intel Corporation.
 * SPDX-License-Identifier: Apache-2.0
 *
 * @test @b eigen_svd_cdouble
 * @parblock
 * This piece of code aims to stress test the FMA execution units of
 * the CPU, among others, by repetitively solving the singular value
 * decomposition problem on given input matrices, which involve a lot
 * of matrix multiplication operations underneath.
 *
 * The logic comes from the 3rd party library Eigen. Initialization
 * computes golden output matrices "U" and "V" from captured inputs.
 * Worker results are compared against those golden matrices and
 * against each worker's first results on subsequent iterations.
 *
 * This particular version of the Eigen SVD tests go for double
 * precision, complex numbers input matrices and the (divide and
 * conquer) bi-diagonalization SVD algorithm.
 *
 * @endparblock
 */

#include "sandstone_eigen_common.h"

using namespace Eigen;

typedef Matrix < std::complex <double >, Dynamic, Dynamic > Mat;
typedef Eigen::BDCSVD < Mat > SVD;

#define M_DIM 300               // weird dim on purpose

using eigen_svd_cdouble_test = EigenSVDTest<SVD, M_DIM>;
DECLARE_TEST(eigen_svd_cdouble, "Eigen SVD (Singular Value Decomposition) solving payload, which issues a bunch of matrix multiplies underneath, now operating on std::complex<double>")
  .groups = DECLARE_TEST_GROUPS(&group_math),
  .test_init = eigen_svd_cdouble_test::init,
  .test_run = eigen_svd_cdouble_test::run,
  .minimum_cpu = cpu_skylake_avx512, // just because (Eigen supports
                                     // that anyway)
  .fracture_loop_count = 5,
  .quality_level = TEST_QUALITY_PROD,
  .crosscheck = CrossCheckGolden,
END_DECLARE_TEST
