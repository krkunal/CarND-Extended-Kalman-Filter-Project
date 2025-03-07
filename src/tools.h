#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

class Tools {
 public:
  /**
   * Constructor.
   */
  Tools();

  /**
   * Destructor.
   */
  virtual ~Tools();

  /**
   * A helper method to calculate RMSE.
   */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, 
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
   * A helper method to calculate Jacobians.
   */
  Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state, const Eigen::MatrixXd& Hj_default);

  /**
   * A helper method to map from the cartesian to polar coordinate
   */
  Eigen::VectorXd MapCartesianToPolar(const Eigen::VectorXd& x_state);

  /**
   * Normalize the value of phi between +/- Pi
   */ 
  void NormalizePhi(double& phi);

};

#endif  // TOOLS_H_
