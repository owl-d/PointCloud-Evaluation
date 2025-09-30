#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/sample_consensus_prerejective.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>


typedef pcl::PointXYZ point;
typedef pcl::PointCloud<point> pCloud;
typedef pcl::PointCloud<point>::Ptr pCloudPtr;

typedef pcl::Normal normal;
typedef pcl::PointCloud<normal> nCloud;
typedef pcl::PointCloud<normal>::Ptr nCloudPtr;

typedef pcl::FPFHSignature33 DescriptorT;
typedef pcl::PointCloud<DescriptorT> dCloud;
typedef pcl::PointCloud<DescriptorT>::Ptr dCloudPtr;

typedef std::vector<Eigen::Vector3f> v3f;

struct PRF {
    double precision; // [0,1]
    double recall;    // [0,1]
    double fscore;    // [0,1]
};

struct RMSE {
    double P2G = 0.0;
    double G2P = 0.0;
    double sym = 0.0;

    size_t N_P2G = 0;
    size_t N_G2P = 0;
};

struct Chamfer {
    double P2G;
    double G2P;
    double L1_avg;
    double L2sq_avg;
    double RMSE_sym;
};

bool loadCloud(std::string& path, pCloudPtr cloud);
pCloudPtr voxelDownsample(pCloudPtr inCloud, float voxel_size);

nCloudPtr estimateNormals(const pCloudPtr& cloud);
v3f estimateNormals_K(const pCloudPtr& cloud, int K=5);
v3f estimateNormals_radius(const pCloudPtr& cloud, double radius, int fallbackK=5, bool fill_missing_with_K=true);
v3f estimateNormals_radius_auto(const pCloudPtr& cloud, double multiplier = 3.0, int k_for_spacing = 6, int fallbackK = 5, bool fill_missing_with_K = true);

std::vector<double> nnDistances(const pCloudPtr& src, const pcl::KdTreeFLANN<point>& kdt_Target);
std::vector<double> nnPerpDistances(const pCloudPtr& src, const pcl::KdTreeFLANN<point>& kdt_trg, const pCloudPtr& trg, const v3f& ntrg);
double estimate_nn_spacing(const pCloudPtr& cloud, int k_for_spacing = 6);

double mean(const std::vector<double>& a);
double mean_sq(const std::vector<double>& a);
double percentile(std::vector<double> a, double q01_to_99);
double rmse_from_dist_sq_mean(double mean_sq);