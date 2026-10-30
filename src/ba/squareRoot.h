#pragma once
#include <Eigen/Dense>

namespace superVO::ba {
struct LinearSystem {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
};
// Eliminate leading nuisance columns by orthogonal projection. Rank deficiency
// is handled explicitly; no damping or artificial information enters a prior.
inline LinearSystem eliminate(const Eigen::MatrixXd& J, const Eigen::VectorXd& r, int columns) {
    if (columns == 0)
        return {J, r};
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J.leftCols(columns));
    qr.setThreshold(1e-8);
    const int rank = qr.rank(), keep = J.rows() - rank;
    const Eigen::MatrixXd reduced = qr.householderQ().adjoint() * J.rightCols(J.cols() - columns);
    const Eigen::VectorXd residual = qr.householderQ().adjoint() * r;
    return {reduced.bottomRows(keep), residual.tail(keep)};
}
inline LinearSystem compress(const Eigen::MatrixXd& J, const Eigen::VectorXd& r) {
    if (!J.rows() || !J.cols())
        return {Eigen::MatrixXd(0, J.cols()), Eigen::VectorXd(0)};
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(J);
    qr.setThreshold(1e-10);
    const int rank = qr.rank();
    Eigen::MatrixXd R = qr.matrixR().topRows(rank);
    R.triangularView<Eigen::StrictlyLower>().setZero();
    R = R * qr.colsPermutation().transpose();
    const Eigen::VectorXd transformed = qr.householderQ().adjoint() * r;
    return {R, transformed.head(rank)};
}
}
