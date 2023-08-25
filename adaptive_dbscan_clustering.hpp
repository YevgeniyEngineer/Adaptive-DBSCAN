#ifndef ADAPTIVE_DBSCAN_CLUSTERING
#define ADAPTIVE_DBSCAN_CLUSTERING

#include "adaptive_dbscan_index.hpp"
#include "adaptive_dbscan_pointcloud.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <nanoflann.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace clustering
{
namespace adaptive_dbscan_labels
{
inline constexpr std::int32_t UNDEFINED = -2;
inline constexpr std::int32_t NOISE = -1;
} // namespace adaptive_dbscan_labels

template <typename CoordinateType, std::size_t MaxPoints = 150'000U> class AdaptiveDBSCANClustering
{
    using AdaptiveDBSCANIndexType = AdaptiveDBSCANIndex<CoordinateType>;

  public:
    AdaptiveDBSCANClustering(const AdaptiveDBSCANClustering &) = delete;
    AdaptiveDBSCANClustering &operator=(const AdaptiveDBSCANClustering &) = delete;
    AdaptiveDBSCANClustering(AdaptiveDBSCANClustering &&) = delete;
    AdaptiveDBSCANClustering &operator=(AdaptiveDBSCANClustering &&) = delete;

    AdaptiveDBSCANClustering(CoordinateType distance_multiplicative_factor = 0.01,
                             CoordinateType min_distance_threshold = 0.01, CoordinateType max_distance_threshold = 1.0,
                             std::uint32_t min_cluster_size = 3)
        : distance_multiplicative_factor_squared_{distance_multiplicative_factor * distance_multiplicative_factor},
          min_distance_threshold_squared_{min_distance_threshold * min_distance_threshold},
          max_distance_threshold_squared_{max_distance_threshold * max_distance_threshold},
          min_cluster_size_{min_cluster_size}, search_parameters_{}, index_adaptor_{nullptr}
    {
        labels_.reserve(MaxPoints);
        neighbours_.reserve(MaxPoints);
        inner_neighbours_.reserve(MaxPoints);
        scaled_search_distances_.reserve(MaxPoints);
    }

    /// @brief Update KDTree index with new points
    void createIndex(const typename AdaptiveDBSCANIndexType::PointCloudType &point_cloud)
    {
        // Create a new adaptor
        index_adaptor_.reset(new AdaptiveDBSCANIndexType(point_cloud));
    }

    /// @brief Get labels of each of the points
    inline const std::vector<std::int32_t> &labels() const noexcept
    {
        return labels_;
    }

    /// @brief Return a map of [label, indices] for each cluster, excluding noise point
    std::unordered_map<std::int32_t, std::vector<std::uint32_t>> getClusterMap()
    {
        std::unordered_map<std::int32_t, std::vector<std::uint32_t>> cluster_map;
        for (std::uint32_t i = 0; i < labels_.size(); ++i)
        {
            const auto &label = labels_[i];

            if (label >= 0)
            {
                cluster_map[label].push_back(i);
            }
        }
        return cluster_map;
    }

    /// @brief Adaptive DBSCAN
    void formClusters()
    {
        labels_.clear();

        if (index_adaptor_ == nullptr)
        {
            throw std::runtime_error("KDTree index was not updated with new points!");
        }

        if (index_adaptor_->pointCloud().coordinates.empty())
        {
            // Nothing to cluster
            return;
        }

        const auto &coordinates = index_adaptor_->pointCloud().coordinates;
        const auto &index = index_adaptor_->index();
        const std::uint32_t number_of_points = coordinates.size() / 3;

        // Set all initial label to UNDEFINED
        labels_.resize(number_of_points);
        for (auto &label : labels_)
        {
            label = adaptive_dbscan_labels::UNDEFINED;
        }

        // Set scaled distances for every point
        scaled_search_distances_.reserve(number_of_points);
        for (std::uint32_t i = 0U; i < coordinates.size(); i += 3)
        {
            const CoordinateType &x = coordinates[i];
            const CoordinateType &y = coordinates[i + 1];
            const CoordinateType &z = coordinates[i + 2];

            scaled_search_distances_.push_back(
                std::clamp(distance_multiplicative_factor_squared_ * (x * x + y * y + z * z),
                           min_distance_threshold_squared_, max_distance_threshold_squared_));
        }

        // Initial cluster label
        std::int32_t label = adaptive_dbscan_labels::NOISE;

        // Iterate over each point
        for (std::uint32_t i = 0U; i < number_of_points; ++i)
        {
            // Check if label is not undefined
            auto &label_i = labels_[i];
            if (label_i != adaptive_dbscan_labels::UNDEFINED)
            {
                continue;
            }

            // Find nearest neighbours within the adaptive search radius
            neighbours_.clear();

            // Recalculate the adaptive radius threshold
            const auto number_of_neighbours =
                index.radiusSearch(&coordinates[i * 3U], scaled_search_distances_[i], neighbours_, search_parameters_);

            // Check if the noise point
            if (number_of_neighbours < min_cluster_size_)
            {
                label_i = adaptive_dbscan_labels::NOISE;
                continue;
            }

            // Set the next cluster label
            ++label;

            // Label the initial point
            label_i = label;

            // Iterate over all neighbours, excluding the point itself
            for (std::uint32_t j = 1U; j < neighbours_.size(); ++j)
            {
                // Get the current label of the neighbour point
                const std::uint32_t index_j = neighbours_[j].first;
                auto &label_j = labels_[index_j];

                // Check if the label is a noise point
                if (label_j == adaptive_dbscan_labels::NOISE)
                {
                    // Assign this point to a new cluster
                    label_j = label;
                    continue;
                }

                // Check if previously processed
                if (label_j != adaptive_dbscan_labels::UNDEFINED)
                {
                    continue;
                }

                // Label the neighbour point
                label_j = label;

                // Find neighbours of this neighbour
                inner_neighbours_.clear();

                const auto number_of_inner_neighbours =
                    index.radiusSearch(&coordinates[index_j * 3U], scaled_search_distances_[index_j], inner_neighbours_,
                                       search_parameters_);

                // Density check, if the point is a core point
                if (number_of_inner_neighbours >= min_cluster_size_)
                {
                    // Add new neighbours to the seed set, excluding the point itself
                    for (std::uint32_t k = 1U; k < inner_neighbours_.size(); ++k)
                    {
                        // Override the label
                        labels_[inner_neighbours_[k].first] = label;
                    }
                }
            }
        }
    }

  private:
    nanoflann::SearchParams search_parameters_;
    std::unique_ptr<AdaptiveDBSCANIndexType> index_adaptor_;
    CoordinateType distance_multiplicative_factor_squared_;
    CoordinateType min_distance_threshold_squared_;
    CoordinateType max_distance_threshold_squared_;
    std::uint32_t min_cluster_size_;
    std::vector<std::int32_t> labels_;
    std::vector<std::pair<std::uint32_t, CoordinateType>> neighbours_;
    std::vector<std::pair<std::uint32_t, CoordinateType>> inner_neighbours_;
    std::vector<CoordinateType> scaled_search_distances_;
};
} // namespace clustering

#endif // ADAPTIVE_DBSCAN_CLUSTERING