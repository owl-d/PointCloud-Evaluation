#pragma once

#include "util.h"

pCloudPtr ICP(pCloudPtr srcCloud, pCloudPtr trgCloud);
pCloudPtr GICP(pCloudPtr& src, pCloudPtr& trg);

dCloudPtr computeFPFH(const pCloudPtr& cloud, const nCloudPtr& normals);
pCloudPtr feature_matching(pCloudPtr srcCloud, pCloudPtr trgCloud);