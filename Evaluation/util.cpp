#include "util.h"

bool loadCloud(std::string& path, pCloudPtr cloud) {
    cloud->clear();
    if (path.size() >= 4) {
        std::string ext = path.substr(path.size() - 4);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".pcd")
            return pcl::io::loadPCDFile<point>(path, *cloud) == 0;
        if (ext == ".ply")
            return pcl::io::loadPLYFile<point>(path, *cloud) == 0;
        if (ext == ".obj")
            return pcl::io::loadOBJFile<point>(path, *cloud) == 0;
    }
    // fallback: try all
    if (pcl::io::loadPCDFile<point>(path, *cloud) == 0) return true;
    if (pcl::io::loadPLYFile<point>(path, *cloud) == 0) return true;
    if (pcl::io::loadOBJFile<point>(path, *cloud) == 0) return true;
    std::cerr << "[Error] Unsupported or failed to load: " << path << "\n";
    return false;
}

pCloudPtr voxelDownsample(pCloudPtr inCloud, float voxel_size){
    if (inCloud->empty()) {
        std::cerr << "[VOXEL] Input cloud is empty!" << std::endl;
    }

    pCloudPtr filteredCloud(new pCloud);
    *filteredCloud = *inCloud;

    pcl::VoxelGrid<point> voxel_filter;
    voxel_filter.setInputCloud(inCloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_filter.filter(*filteredCloud);

    std::cerr << "[VOXEL] voxel downsampling " << inCloud->size() << " -> " << filteredCloud->size() << endl;

    return filteredCloud;
}

nCloudPtr estimateNormals(const pCloudPtr& cloud) {
    nCloudPtr normals(new nCloud);
    pcl::NormalEstimation<point, normal> ne;
    pcl::search::KdTree<point>::Ptr tree(new pcl::search::KdTree<point>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    //ne.setRadiusSearch(10);
    ne.setKSearch(20);
    ne.compute(*normals);
    return normals;
}

// normals & point-to-plane
v3f estimateNormals_K(const pCloudPtr& cloud, int K) {
    std::vector<Eigen::Vector3f> normals;
    normals.resize(cloud->size(), Eigen::Vector3f::Zero());

    pcl::NormalEstimation<point, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<point>::Ptr tree(new pcl::search::KdTree<point>());
    ne.setSearchMethod(tree);
    ne.setKSearch(std::max(3, K));

    pcl::PointCloud<pcl::Normal>::Ptr n(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*n);

    for (size_t i = 0; i < cloud->size() && i < n->size(); ++i) {
        const auto& nn = n->points[i];
        Eigen::Vector3f v(nn.normal_x, nn.normal_y, nn.normal_z);
        float l = v.norm();
        if (std::isfinite(l) && l > 1e-6f) normals[i] = v / l; // unit normal
    }
    return normals;
}

v3f estimateNormals_radius(const pCloudPtr& cloud, double radius, int fallbackK, bool fill_missing_with_K){
    v3f normals(cloud->size(), Eigen::Vector3f::Zero());
    if (!cloud || cloud->empty() || radius <= 0.0) return normals;

    pcl::NormalEstimation<point, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<point>::Ptr tree(new pcl::search::KdTree<point>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    pcl::PointCloud<pcl::Normal>::Ptr n(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*n);

    std::vector<char> ok(n->size(), 0);
    for (size_t i = 0; i < cloud->size() && i < n->size(); ++i) {
        Eigen::Vector3f v(n->points[i].normal_x, n->points[i].normal_y, n->points[i].normal_z);
        float L = v.norm();
        if (std::isfinite(L) && L > 1e-6f) {
            normals[i] = v / L;
            ok[i] = 1;
        }
    }

    if (fill_missing_with_K) {
        bool need_fallback = false;
        for (char f : ok) { if (!f) { need_fallback = true; break; } }
        if (need_fallback) {
            pcl::NormalEstimation<point, pcl::Normal> neK;
            neK.setInputCloud(cloud);
            neK.setSearchMethod(tree);
            neK.setKSearch(std::max(3, fallbackK));
            pcl::PointCloud<pcl::Normal>::Ptr nK(new pcl::PointCloud<pcl::Normal>());
            neK.compute(*nK);

            for (size_t i = 0; i < cloud->size() && i < nK->size(); ++i) if (!ok[i]) {
                Eigen::Vector3f v(nK->points[i].normal_x, nK->points[i].normal_y, nK->points[i].normal_z);
                float L = v.norm();
                if (std::isfinite(L) && L > 1e-6f) normals[i] = v / L;
            }
        }
    }

    return normals;
}

v3f estimateNormals_radius_auto(const pCloudPtr& cloud, double multiplier, int k_for_spacing, int fallbackK, bool fill_missing_with_K){
    double spacing = estimate_nn_spacing(cloud, k_for_spacing);
    double radius = (spacing > 0.0 ? multiplier * spacing : 0.0);
    if (radius <= 0.0) {
        return estimateNormals_radius(cloud, 0.0, fallbackK, fill_missing_with_K);
    }
    cout << "Determined auto radius: " << radius << endl;
    return estimateNormals_radius(cloud, radius, fallbackK, fill_missing_with_K);
}

std::vector<double> nnDistances(const pCloudPtr& src, const pcl::KdTreeFLANN<point>& kdt_Target) {
    std::vector<double> out; out.reserve(src->size());
    std::vector<int> idx(1);
    std::vector<float> sqdist(1);
    for (const auto& p : src->points) {
        if (kdt_Target.nearestKSearch(p, 1, idx, sqdist) > 0) {
            out.push_back(std::sqrt(static_cast<double>(sqdist[0])));
        }
        else {
            out.push_back(std::numeric_limits<double>::infinity());
        }
    }
    return out;
}

std::vector<double> nnPerpDistances(const pCloudPtr& src, const pcl::KdTreeFLANN<point>& kdt_trg,
    const pCloudPtr& trg, const v3f& ntrg) // normals of target, same size as Tgt
{
    std::vector<double> out; out.reserve(src->size());
    std::vector<int> idx(1);
    std::vector<float> sqdist(1);

    for (const auto& s : src->points) {
        double d_perp = std::numeric_limits<double>::infinity();
        if (kdt_trg.nearestKSearch(s, 1, idx, sqdist) > 0) {
            const int j = idx[0];
            const auto& t = trg->points[j];
            Eigen::Vector3f n = (j < (int)ntrg.size() ? ntrg[j] : Eigen::Vector3f::Zero());
            Eigen::Vector3f diff(s.x - t.x, s.y - t.y, s.z - t.z);
            if (n.norm() > 1e-6f) d_perp = std::abs(n.dot(diff));                // point-to-plane
            else                  d_perp = std::sqrt((double)sqdist[0]);         // fallback: p2p
        }
        out.push_back(d_perp);
    }
    return out;
}

double estimate_nn_spacing(const pCloudPtr& cloud, int k_for_spacing) {
    if (!cloud || cloud->size() < 2) return 0.0;
    pcl::KdTreeFLANN<point> kdt; kdt.setInputCloud(cloud);

    std::vector<double> dists; dists.reserve(cloud->size());
    std::vector<int> idx(k_for_spacing);
    std::vector<float> sq(k_for_spacing);

    for (const auto& p : cloud->points) {
        if (kdt.nearestKSearch(p, k_for_spacing, idx, sq) >= 2) {
            double d = std::sqrt(static_cast<double>(sq[1]));
            if (std::isfinite(d)) dists.push_back(d);
        }
    }
    if (dists.empty()) return 0.0;
    size_t mid = dists.size() / 2;
    std::nth_element(dists.begin(), dists.begin() + mid, dists.end());
    return dists[mid];
}

double mean(const std::vector<double>& a) {
    if (a.empty()) return 0.0;
    long double s = 0.0L;
    for (double v : a) s += v;
    return static_cast<double>(s / a.size());
}

double mean_sq(const std::vector<double>& a) {
    if (a.empty()) return 0.0;
    long double s = 0.0L;
    for (double v : a) s += v * v;
    return static_cast<double>(s / a.size());
}

double percentile(std::vector<double> a, double q01_to_99) {
    if (a.empty()) return 0.0;
    if (q01_to_99 <= 0) q01_to_99 = 0;
    if (q01_to_99 >= 100) q01_to_99 = 100;
    std::sort(a.begin(), a.end());
    double rank = (q01_to_99 / 100.0) * (a.size() - 1);
    size_t i = static_cast<size_t>(std::floor(rank));
    size_t j = static_cast<size_t>(std::ceil(rank));
    double w = rank - i;
    return (1.0 - w) * a[i] + w * a[j];
}