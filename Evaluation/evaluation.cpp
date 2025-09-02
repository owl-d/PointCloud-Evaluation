#include "evaluation.h"


// Precision/Recall/F-score@tau calculation point-to-point
PRF computePRF_at_tau_point2point(const pCloudPtr& P, const pCloudPtr& G, double tau) {
    if (P->empty() || G->empty()) {
        return { 0.0, 0.0, 0.0 };
    }

    pcl::KdTreeFLANN<point> kdt_P2G, kdt_G2P;
    kdt_P2G.setInputCloud(G);
    kdt_G2P.setInputCloud(P);

    std::vector<int> idx(1);
    std::vector<float> sqdist(1);

    // Precision: P->G
    size_t hit_P = 0;
    for (const auto& p : P->points) {
        if (kdt_P2G.nearestKSearch(p, 1, idx, sqdist) > 0) {
            double d = std::sqrt(static_cast<double>(sqdist[0]));
            if (d < tau) ++hit_P;
        }
    }
    double precision = static_cast<double>(hit_P) / static_cast<double>(P->size());

    // Recall: G->P
    size_t hit_G = 0;
    for (const auto& g : G->points) {
        if (kdt_G2P.nearestKSearch(g, 1, idx, sqdist) > 0) {
            double d = std::sqrt(static_cast<double>(sqdist[0]));
            if (d < tau) ++hit_G;
        }
    }
    double recall = static_cast<double>(hit_G) / static_cast<double>(G->size());

    double fscore = (precision + recall > 0.0)
        ? (2.0 * precision * recall) / (precision + recall)
        : 0.0;

    return { precision, recall, fscore };
}

PRF computePRF_at_tau_point2point2(const std::vector<double>& d_P2G,
    const std::vector<double>& d_G2P, double tau) {
    size_t hitP = 0, hitG = 0;
    for (double d : d_P2G) if (d < tau) ++hitP;
    for (double d : d_G2P) if (d < tau) ++hitG;
    double P = (d_P2G.empty() ? 0.0 : static_cast<double>(hitP) / d_P2G.size());
    double R = (d_G2P.empty() ? 0.0 : static_cast<double>(hitG) / d_G2P.size());
    double F = (P + R > 0) ? (2.0 * P * R) / (P + R) : 0.0;
    return { P,R,F };
}

// Precision/Recall/F-score@tau calculation point-to-plane
PRF computePRF_at_tau_point2plane(const pCloudPtr& P, const pCloudPtr& G,
    const v3f& nP, const v3f& nG, double tau) {
    if (P->empty() || G->empty()) return { 0,0,0 };

    pcl::KdTreeFLANN<point> kdtG; kdtG.setInputCloud(G);
    pcl::KdTreeFLANN<point> kdtP; kdtP.setInputCloud(P);

    std::vector<int> idx(1);
    std::vector<float> sqdist(1);

    // Precision: P -> G (use G's normals)
    size_t hitP = 0;
    for (size_t i = 0; i < P->size(); ++i) {
        const auto& p = P->points[i];
        if (kdtG.nearestKSearch(p, 1, idx, sqdist) > 0) {
            const int j = idx[0];
            const auto& g = G->points[j];
            Eigen::Vector3f n = (j < nG.size() ? nG[j] : Eigen::Vector3f::Zero());
            Eigen::Vector3f diff(p.x - g.x, p.y - g.y, p.z - g.z);
            double d_perp;
            if (n.norm() > 1e-6f) d_perp = std::abs(n.dot(diff));
            else                  d_perp = std::sqrt(static_cast<double>(sqdist[0])); // fallback: p2p
            if (d_perp < tau) ++hitP;
        }
    }
    double Precision = static_cast<double>(hitP) / P->size();

    // Recall: G -> P (use P's normals)
    size_t hitG = 0;
    for (size_t j = 0; j < G->size(); ++j) {
        const auto& g = G->points[j];
        if (kdtP.nearestKSearch(g, 1, idx, sqdist) > 0) {
            const int iP = idx[0];
            const auto& p = P->points[iP];
            Eigen::Vector3f n = (iP < nP.size() ? nP[iP] : Eigen::Vector3f::Zero());
            Eigen::Vector3f diff(g.x - p.x, g.y - p.y, g.z - p.z);
            double d_perp;
            if (n.norm() > 1e-6f) d_perp = std::abs(n.dot(diff));
            else                  d_perp = std::sqrt(static_cast<double>(sqdist[0])); // fallback
            if (d_perp < tau) ++hitG;
        }
    }
    double Recall = static_cast<double>(hitG) / G->size();
    double F = (Precision + Recall > 0) ? (2.0 * Precision * Recall) / (Precision + Recall) : 0.0;
    return { Precision, Recall, F };
}

Chamfer chamfer_point_to_point(const std::vector<double>& dP2G, const std::vector<double>& dG2P){
    double mP = mean(dP2G);
    double mG = mean(dG2P);
    double mP2 = mean_sq(dP2G);
    double mG2 = mean_sq(dG2P);

    Chamfer out;
    out.P2G = mP;
    out.G2P = mG;
    out.L1_avg = 0.5 * (mP + mG);
    out.L2sq_avg = 0.5 * (mP2 + mG2);
    out.RMSE_sym = std::sqrt(out.L2sq_avg);
    return out;
}

Chamfer chamfer_point_to_plane(const std::vector<double>& dP2G_perp, const std::vector<double>& dG2P_perp){
    auto m = [](const std::vector<double>& a) { long double s = 0; for (double v : a) s += v; return (a.empty() ? 0.0 : (double)(s / a.size())); };
    auto m2 = [](const std::vector<double>& a) { long double s = 0; for (double v : a) s += v * v; return (a.empty() ? 0.0 : (double)(s / a.size())); };

    double mP = m(dP2G_perp);
    double mG = m(dG2P_perp);
    double mP2 = m2(dP2G_perp);
    double mG2 = m2(dG2P_perp);

    Chamfer out;
    out.P2G = mP;
    out.G2P = mG;
    out.L1_avg = 0.5 * (mP + mG);
    out.L2sq_avg = 0.5 * (mP2 + mG2);
    out.RMSE_sym = std::sqrt(out.L2sq_avg);
    return out;
}