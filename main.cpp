#include "adaptive_dbscan_clustering.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_map>

int main()
{
    static constexpr std::uint32_t NUMBER_OF_POINTS = 100'000U;
    static constexpr std::uint32_t NUMBER_OF_COORDINATES = 3U * NUMBER_OF_POINTS;

    // Create random points
    using CoordinateType = float;
    using AdaptiveDBSCANType = clustering::AdaptiveDBSCANClustering<CoordinateType>;
    using AdaptiveDBSCANPointCloudType = clustering::AdaptiveDBSCANPointCloud<CoordinateType>;
    using PointCloudType = AdaptiveDBSCANPointCloudType::PointCloud;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    PointCloudType coordinates;
    coordinates.reserve(NUMBER_OF_COORDINATES);

    for (std::size_t i = 0; i < NUMBER_OF_COORDINATES; ++i)
    {
        coordinates.push_back(dis(gen));
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Construct DBScan class
    AdaptiveDBSCANType dbscan{0.5F, 0.02F, 1.0F, 3U};

    // Create point cloud adapter
    const AdaptiveDBSCANPointCloudType point_cloud{coordinates};

    // Create KDTree index
    dbscan.createIndex(point_cloud);

    // Cluster points
    dbscan.formClusters();

    // Get clusters
    const auto &labels = dbscan.labels();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time [ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << std::endl;

    // Iterate through points and combine
    std::unordered_map<std::int32_t, std::vector<std::uint32_t>> labels_map;
    for (std::uint32_t i = 0; i < labels.size(); ++i)
    {
        const auto &label = labels[i];

        if (label >= 0)
        {
            labels_map[label].push_back(i);
        }
    }

    // Print number of clusters
    std::cout << "Number of clusters: " << labels_map.size() << std::endl;

    return 0;
}