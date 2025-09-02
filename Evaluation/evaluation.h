#pragma once

#include "util.h"

PRF computePRF_at_tau_point2point(const pCloudPtr& P, const pCloudPtr& G, double tau);
PRF computePRF_at_tau_point2point2(const std::vector<double>& d_P2G, const std::vector<double>& d_G2P, double tau);
PRF computePRF_at_tau_point2plane(const pCloudPtr& P, const pCloudPtr& G, const v3f& nP, const v3f& nG, double tau);
Chamfer chamfer_point_to_point(const std::vector<double>& dP2G, const std::vector<double>& dG2P);
Chamfer chamfer_point_to_plane(const std::vector<double>& dP2G_perp, const std::vector<double>& dG2P_perp);