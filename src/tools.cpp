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
    // compute the Jacobian matrix
    Hj << px / (sqrt(px*px + py*py)), py / (sqrt(px*px + py*py)), 0, 0, 
          -py / (px*px + py*py), px / (px*px + py*py), 0, 0,
          py*(vx*py - vy*px) / pow(px*px + py*py, 3/2), px*(vy*px - vx*py) / pow(px*px + py*py, 3/2), px / (sqrt(px*px + py*py)), py / (sqrt(px*px + py*py));   
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
  float c1 = sqrt(px*px + py*py);
  float c2 = px*vx + py*vy;
  // check division by zero
  if (fabs(c1) < 0.0001) {
    cout << "MapCartesianToPolar () - Error - Division by Zero" << endl;
    hx << 0, 
          0, 
          0;
  }
  else {
    hx << c1, 
          atan2(py, px), 
          c2 / c1;
  }
  return hx;
}

double Tools::NormalizePhi(double& phi) {
  // Normalize phi component in y
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
  return phi;
}