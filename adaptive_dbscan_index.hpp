#ifndef ADAPTIVE_DBSCAN_INDEX_HPP
#define ADAPTIVE_DBSCAN_INDEX_HPP

#include "adaptive_dbscan_pointcloud.hpp"
#include <nanoflann.hpp>

namespace clustering
{
template <typename CoordinateType> class AdaptiveDBSCANIndex
{
  public:
    using PointCloudType = AdaptiveDBSCANPointCloud<CoordinateType>;
    using IndexType = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<CoordinateType, PointCloudType>,
                                                          PointCloudType, 3>;

    explicit AdaptiveDBSCANIndex(const PointCloudType &point_cloud)
        : point_cloud_{point_cloud}, index_{3 /* dim */, point_cloud,
                                            nanoflann::KDTreeSingleIndexAdaptorParams{10 /* max leaf */}}
    {
    }

    inline const IndexType &index() const noexcept
    {
        return index_;
    }

    inline const PointCloudType &pointCloud() const noexcept
    {
        return point_cloud_;
    }

  private:
    const PointCloudType &point_cloud_;
    IndexType index_;
};
} // namespace clustering

#endif // ADAPTIVE_DBSCAN_INDEX_HPP