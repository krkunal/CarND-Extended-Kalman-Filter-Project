#include "tools.h"
#include <iostream>
#include <stdexcept>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::invalid_argument;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() == 0) {
      throw invalid_argument("estimation vector size is zero.");
  }
  else if (estimations.size() != ground_truth.size()) {
      throw invalid_argument("estimation and ground truth vectors should be of same size!");
  }
  // accumulate squared residuals
  for (int i=0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    rmse.array() += residual.array()*residual.array();
  }

  // calculate the mean
  rmse.array() /= estimations.size();
  // calculate the squared root
  rmse = rmse.array().sqrt();    
  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state, const Eigen::MatrixXd& Hj_default) {
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if (px == 0 && py == 0) {
      cerr << "Division by zero." << endl;
      Hj = Hj_default;
  }
  else {
    double norm = px*px + py*py;
    double sqrt_norm = sqrt(norm);
    double pxv =  (vy*px - vx*py) / pow(norm, 3/2); 
    // compute the Jacobian matrix
    Hj << px / sqrt_norm, py / sqrt_norm, 0, 0, 
          -py / norm, px / norm, 0, 0,
          -py * pxv, px * pxv, px / sqrt_norm, py / sqrt_norm;
  }
  return Hj; 
}

VectorXd Tools::MapCartesianToPolar(const VectorXd& x_state) {
  VectorXd hx(3);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute repeated terms
  float sqrt_norm = sqrt(px*px + py*py);
  float pdotv = px*vx + py*vy;
  // check division by zero
  if (fabs(sqrt_norm) < 0.0001) {
    cout << "MapCartesianToPolar () - Error - Division by Zero" << endl;
    hx << 0, 
          0, 
          0;
  }
  else {
    hx << sqrt_norm, 
          atan2(py, px), 
          pdotv / sqrt_norm;
  }
  return hx;
}

void Tools::NormalizePhi(double& phi) {
  // Normalize phi component in y E [-PI, PI]
  while (phi > M_PI | phi < - M_PI)
  {
    if (phi > M_PI)
    {
      phi = 2*M_PI - phi;
    }
    else {
      phi += 2*M_PI;
    }
    
  }
}