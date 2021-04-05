#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_; // 4x1
  VectorXd y = z - z_pred; // 4x1
  MatrixXd Ht = H_.transpose(); // 4x4
  MatrixXd S = H_ * P_ * Ht + R_; // 4x4
  MatrixXd Si = S.inverse(); // 4x4
  MatrixXd PHt = P_ * Ht; // 4x4
  MatrixXd K = PHt * Si; // 4x4

  //new estimate
  x_ = x_ + (K * y); // 4x1
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_; // 4x4
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Perform the mapping from cartesian to polar coords.
  VectorXd hx_ = tools.MapCartesianToPolar(x_);
  VectorXd y = z - hx_; // 3x1
  // Normalize the phi component of y.
  tools.NormalizePhi(y(1));
  // get the Jacobian Matrix Hj
  MatrixXd Hj = tools.CalculateJacobian(x_, H_); // 3x4  
  MatrixXd Hjt = Hj.transpose(); // 4x3 
  MatrixXd S = Hj * P_ * Hjt + R_; // 3x3
  MatrixXd Si = S.inverse(); // 3x3
  MatrixXd PHt = P_ * Hjt; // 4x3
  MatrixXd K = PHt * Si; // 4x3

  //new estimate
  x_ = x_ + (K * y); // 4x1
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * Hj) * P_; // 4x4
}
