#include "registration.h"

pCloudPtr ICP(pCloudPtr srcCloud, pCloudPtr trgCloud) {

    // ICP
    pcl::IterativeClosestPoint<point, point> icp;

    // Parameter Set -------------------------------------------------------------
    icp.setMaxCorrespondenceDistance(5.0);    // def=5.0
    icp.setMaximumIterations(200);            // def=50
    icp.setTransformationEpsilon(1e-6);       // def=1e-6
    icp.setEuclideanFitnessEpsilon(1e-6);     // def=-1.79769e+308
    icp.setRANSACOutlierRejectionThreshold(0.01); // def=0.05
    icp.setRANSACIterations(0);               // def=0 (disable RANSAC)
    //----------------------------------------------------------------------------


    pCloudPtr aligned(new pCloud);

    if (srcCloud->points.empty() or trgCloud->points.empty()) {
        std::cout << "[ICP] Point Cloud is Empty !" << std::endl;
        return trgCloud;
    }

    icp.setInputSource(srcCloud);
    icp.setInputTarget(trgCloud);
    icp.align(*aligned);

    if (!icp.hasConverged()) {
        std::cerr << "[ICP] Not converged. Fitness: " << icp.getFitnessScore() << endl;
    }

    cout << "[ICP] Converged. Fitness: " << icp.getFitnessScore() << endl;
    return aligned;
}

pCloudPtr GICP(pCloudPtr& src, pCloudPtr& trg)
{
    pcl::GeneralizedIterativeClosestPoint<point, point> gicp;
    gicp.setInputSource(src);
    gicp.setInputTarget(trg);

    // Parameter Set -------------------------------------------------------------
    // gicp parameter
    gicp.setMaxCorrespondenceDistance(5);		//def=5
    gicp.setCorrespondenceRandomness(10);		//def=20
    gicp.setMaximumOptimizerIterations(200);		//def=20
    gicp.setTranslationGradientTolerance(0.001);	//def=0.01
    gicp.setRotationGradientTolerance(0.001);	//def=0.01
    gicp.setRotationEpsilon(1e-9);				//def=0.02

    // icp parameter
    gicp.setUseReciprocalCorrespondences(true);	//def=false

    // registration parameter
    gicp.setMaximumIterations(50);					//def=200
    gicp.setRANSACIterations(0);					//def=0
    gicp.setRANSACOutlierRejectionThreshold(0.01);	//def=0.05
    gicp.setTransformationEpsilon(1e-5);			//def=0.0005
    gicp.setTransformationRotationEpsilon(1e-5);	//def=0
    gicp.setEuclideanFitnessEpsilon(1e-5);			//def=-1.79769e+308
    //----------------------------------------------------------------------------

    pCloudPtr aligned;
    aligned.reset(new pCloud);
    gicp.align(*aligned);

    if (!gicp.hasConverged()) {
        std::cerr << "[GICP] Not converged. Fitness: " << gicp.getFitnessScore() << endl;
    }

    cout << "[GICP] Converged. Fitness: " << gicp.getFitnessScore() << endl;
    return aligned;
}

dCloudPtr computeFPFH(const pCloudPtr& cloud, const nCloudPtr& normals) {
    if (cloud->size() != normals->size()) {
        std::cerr << "[computeFPFH] ERR : Point and normal cloud size mismatch! "
            << cloud->size() << " vs " << normals->size() << std::endl;
    }

    dCloudPtr fpfh(new dCloud);
    pcl::FPFHEstimation<point, normal, DescriptorT> fpfh_est;
    fpfh_est.setInputCloud(cloud);
    fpfh_est.setInputNormals(normals);
    pcl::search::KdTree<point>::Ptr tree(new pcl::search::KdTree<point>);
    fpfh_est.setSearchMethod(tree);
    fpfh_est.setRadiusSearch(10);
    fpfh_est.compute(*fpfh);
    std::cout << "[computeFPFH] size " << fpfh->size() << std::endl;
    return fpfh;
};

pCloudPtr feature_matching(pCloudPtr srcCloud, pCloudPtr trgCloud) {

    if (srcCloud->points.empty() || trgCloud->points.empty()) {
        std::cout << "[Feature matching] Empty Point Cloud!" << std::endl;
        return trgCloud;
    }

    // normal estimation
    nCloudPtr normals_src = estimateNormals(srcCloud);
    nCloudPtr normals_trg = estimateNormals(trgCloud);

    // FPFH feature extraction 1
    dCloudPtr fpfh_src(new dCloud);
    pcl::FPFHEstimation<point, pcl::Normal, DescriptorT> fpfh_est;
    fpfh_est.setInputCloud(srcCloud);
    fpfh_est.setInputNormals(normals_src);
    pcl::search::KdTree<point>::Ptr tree(new pcl::search::KdTree<point>);
    fpfh_est.setSearchMethod(tree);
    fpfh_est.setRadiusSearch(5);
    fpfh_est.compute(*fpfh_src);

    dCloudPtr fpfh_trg(new dCloud);
    fpfh_est.setInputCloud(trgCloud);
    fpfh_est.setInputNormals(normals_trg);
    fpfh_est.compute(*fpfh_trg);

    // FPFH feature extraction 2
    //std::cout << "[Feature matching] 1 " << std::endl;
    //dCloudPtr fpfh_src = computeFPFH(tfCloud, normals_src);
    //std::cout << "[Feature matching] 2 " << std::endl;
    //dCloudPtr fpfh_trg = computeFPFH(resultCloud, normals_trg);

    if (fpfh_src->empty() || fpfh_trg->empty()) {
        std::cerr << "[Feature matching] Empty FPFH descriptors!" << std::endl;
    }
    std::cout << "[Feature matching] FPFH source size: " << fpfh_src->size() << std::endl;
    std::cout << "[Feature matching] FPFH target size: " << fpfh_trg->size() << std::endl;

    // Feature Matching based RANSAC registration
    pcl::SampleConsensusPrerejective<point, point, DescriptorT> align;
    align.setInputSource(srcCloud);
    align.setSourceFeatures(fpfh_src);
    align.setInputTarget(trgCloud);
    align.setTargetFeatures(fpfh_trg);
    align.setMaximumIterations(100);
    align.setNumberOfSamples(3); // minimum correspondence
    align.setCorrespondenceRandomness(4);
    align.setSimilarityThreshold(0.9f);
    align.setMaxCorrespondenceDistance(5);
    align.setInlierFraction(0.25f);

    pCloudPtr alignedCloud(new pCloud);
    align.align(*alignedCloud);

    if (align.hasConverged()) {
        std::cout << "[Feature matching] converged, Score: " << align.getFitnessScore() << std::endl;
    }
    else {
        std::cout << "[Feature matching] not converged Score: " << align.getFitnessScore() << std::endl;
    }

    //// Fine Registration with ICP
    //pcl::IterativeClosestPoint<point, point> icp;
    //icp.setInputSource(alignedCloud);
    //icp.setInputTarget(resultCloud);
    //pCloudPtr icp_result(new pCloud);
    //icp.align(*icp_result);
    //std::cout << "[Feature matching] ICP has converged: " << icp.hasConverged() << " score: "
    //    << icp.getFitnessScore() << std::endl;

    return alignedCloud;
}