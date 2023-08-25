#ifndef ADAPTIVE_DBSCAN_POINTCLOUD_HPP
#define ADAPTIVE_DBSCAN_POINTCLOUD_HPP

#include <array>
#include <memory>
#include <vector>

namespace clustering
{
/// @brief Point Struct defined for 3 dimensions
template <typename CoordinateType> struct AdaptiveDBSCANPointCloud
{
    using PointCloud = std::vector<CoordinateType>;

    // Container for coordinates in the sequence {x0, y0, z0, x1, y1, z1, ..., xn, yn, zn}
    const PointCloud &coordinates;

    /// @brief Delete default constructor to avoid empty reference
    AdaptiveDBSCANPointCloud() = delete;

    /// @brief Constructor of the point cloud adapter
    /// @param coordinates Input coordinates
    explicit AdaptiveDBSCANPointCloud(const PointCloud &coordinates) : coordinates{coordinates}
    {
    }

    /// @brief Return the number of points in the cloud
    inline std::size_t kdtree_get_point_count() const noexcept
    {
        return coordinates.size() / 3U;
    }

    /// @brief Get a point along the specified dimension
    inline CoordinateType kdtree_get_pt(const std::size_t idx, const std::size_t dim) const noexcept
    {
        // Assuming dim is always correct
        return coordinates[idx * 3U + dim];
    }

    /// @brief Optional bounding box computation
    template <class Bbox> inline bool kdtree_get_bbox(Bbox & /* bb */) const noexcept
    {
        return false;
    }
};
} // namespace clustering

#endif // ADAPTIVE_DBSCAN_POINTCLOUD_HPP