#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>

#include "evaluation.h"
#include "registration.h"
#include "util.h"

using namespace std;

void viewer(pCloudPtr& G, pCloudPtr& P) {
    pcl::visualization::PCLVisualizer::Ptr vis(new pcl::visualization::PCLVisualizer("Clouds"));
    vis->setBackgroundColor(0.05, 0.05, 0.05);
    vis->addCoordinateSystem(0.01);
    //vis->initCameraParameters();

    // GT (green-ish)
    vis->addPointCloud<point>(G, "cloud_G");
    vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.31, 0.78, 0.47, "cloud_G");
    vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cloud_G");

    // Scan (red-ish)
    vis->addPointCloud<point>(P, "cloud_P");
    vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.86, 0.31, 0.31, "cloud_P");
    vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cloud_P");

    while (!vis->wasStopped()) vis->spinOnce(16);
}

int main() {

    string trg_path = "C:/Users/doyu/Desktop/Evaluation/data/top_tf.ply";
    string src_path = "C:/Users/doyu/Desktop/Evaluation/data/result_1.ply";

    // parameter
    vector<double> taus_mm = { 5, 1, 0.5 };
    int normalMode = 3; // 1 kNN 2 radius else radius_auto
    int K = 5;
    double R = 1.0;
    double M = 2.0;


    pCloudPtr G(new pCloud);
    pCloudPtr P(new pCloud);

    if (!loadCloud(trg_path, G)) return 1;
    if (!loadCloud(src_path, P)) return 1;

    // voxel downsampling
    //G = voxelDownsample(G, 1.0);
    //P = voxelDownsample(P, 1.0);

    cout << "\nLoaded:\n";
    cout << "  G (trg) : " << G->size() << " points " << trg_path << endl;
    cout << "  P (src): " << P->size() << " points " << src_path << endl;
    if (G->empty() || P->empty()) {
        cerr << "[Error] One of G or P is empty.\n";
        return 2;
    }
    viewer(G, P);

    // Fine alignment ---------------------------------------------------------
    bool alignment = false;
    if (alignment) {
        cout << "\nAlign...\n";
        pCloudPtr P_aligned(new pCloud);
        P_aligned = ICP(P, G);
        //P_aligned = GICP(P, G);
        //P_aligned = feature_matching(P, G);

        if (P_aligned->empty()) {
            cerr << "[Error] P_aligned cloud is empty.\n";
            return 2;
        }
        else {
            viewer(G, P_aligned);
            P = P_aligned;
        }
    }
    // ------------------------------------------------------------------------

    cout.setf(std::ios::fixed); cout.precision(3);

    pcl::KdTreeFLANN<point> kdtG; kdtG.setInputCloud(G);
    pcl::KdTreeFLANN<point> kdtP; kdtP.setInputCloud(P);

    // ---------- Precompute nearest distances ----------
    std::vector<double> d_P2G = nnDistances(P, kdtG); // |P|
    std::vector<double> d_G2P = nnDistances(G, kdtP); // |G|
                                                      
    v3f nG, nP;
    if (normalMode == 1) {
        // normal estimation with kNN
        nG = estimateNormals_K(G, K);
        nP = estimateNormals_K(P, K);
    }
    else if (normalMode == 2) {
        // normal estimation with radius search
        nG = estimateNormals_radius(G, R, K);
        nP = estimateNormals_radius(P, R, K);
    }
    else {
        // normal estimation with auto radius
        nG = estimateNormals_radius_auto(G, M, 5, K); //0.095*M
        nP = estimateNormals_radius_auto(P, M, 5, K); //0.85*M
    }

    std::vector<double> dP2G_perp = nnPerpDistances(P, kdtG, G, nG);
    std::vector<double> dG2P_perp = nnPerpDistances(G, kdtP, P, nP);

    // ---------- Evaluation --------------------------------
    cout << "\n--- 1) Precision/Recall/F-score point-to-point evaluation ---\n";

    for (double tau_mm : taus_mm) {
        PRF r = computePRF_at_tau_point2point2(d_P2G, d_G2P, tau_mm);
        cout << "tau = " << tau_mm << " mm \n";
        cout << "  Precision@tau : " << r.precision << endl;
        cout << "  Recall@tau    : " << r.recall << endl;
        cout << "  F-score@tau   : " << r.fscore << endl;
    }

    cout << "\n--- 2) Precision/Recall/F-score point-to-plane evaluation ---\n";
    for (double tau_mm : taus_mm) {
        PRF r = computePRF_at_tau_point2plane(P, G, nP, nG, tau_mm);
        cout << "tau = " << tau_mm << " mm \n";
        cout << "  Precision@tau : " << r.precision << endl;
        cout << "  Recall@tau    : " << r.recall << endl;
        cout << "  F-score@tau   : " << r.fscore << endl;
    }

    cout << "\n--- 3) RMSE point-to-point ---\n";
    RMSE r_p2p = computeRMSE_point2point(P, G, 5.0);
    std::cout << "  P->G: " << r_p2p.P2G << " mm  (N=" << r_p2p.N_P2G << ")\n";
    std::cout << "  G->P: " << r_p2p.G2P << " mm  (N=" << r_p2p.N_G2P << ")\n";
    std::cout << "  sym : " << r_p2p.sym << " mm\n";

    cout << "\n--- 4) RMSE point-to-plane ---\n";
    RMSE r_p2pl = computeRMSE_point2plane(P, G, nP, nG, 5.0);
    std::cout << "  P->G: " << r_p2pl.P2G << " mm  (N=" << r_p2pl.N_P2G << ")\n";
    std::cout << "  G->P: " << r_p2pl.G2P << " mm  (N=" << r_p2pl.N_G2P << ")\n";
    std::cout << "  sym : " << r_p2pl.sym << " mm\n";

    cout << "\n--- 5) Chamfer Distance point-to-point ---\n";
    Chamfer c = chamfer_point_to_point(d_P2G, d_G2P);
    std::cout << "  P->G mean    : " << c.P2G << " mm\n";
    std::cout << "  G->P mean    : " << c.G2P << " mm\n";
    std::cout << "  CD-L1 (avg)   : " << c.L1_avg << " mm\n";
    //std::cout << "  CD-L2^2 (avg)  : " << c.L2sq_avg << " mm^2\n";
    std::cout << "  CD-RMSE (sym) : " << c.RMSE_sym << " mm\n";

    cout << "\n--- 6) Chamfer Distance point-to-plane ---\n";
    Chamfer c_perp = chamfer_point_to_plane(dP2G_perp, dG2P_perp);
    std::cout << "  P->G mean    : " << c_perp.P2G << " mm\n";
    std::cout << "  G->P mean    : " << c_perp.G2P << " mm\n";
    std::cout << "  CD-L1 (avg)   : " << c_perp.L1_avg << " mm\n";
    //std::cout << "  CD-L2^2 (avg)  : " << c_perp.L2sq_avg << " mm^2\n";
    std::cout << "  CD-RMSE (sym) : " << c_perp.RMSE_sym << " mm\n";

    cout << "\n--- 7) Hausdorff-95% (robust) ---\n";
    double p95_P2G = percentile(d_P2G, 95.0);
    double p95_G2P = percentile(d_G2P, 95.0);
    double H95 = std::max(p95_P2G, p95_G2P);
    std::cout << "  P->G 95th : " << p95_P2G << " mm\n";
    std::cout << "  G->P 95th : " << p95_G2P << " mm\n";
    std::cout << "  H-95      : " << H95 << " mm\n";


	return 0;
}