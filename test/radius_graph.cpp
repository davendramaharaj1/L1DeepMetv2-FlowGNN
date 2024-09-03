#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Function to calculate the Euclidean distance between two points
double euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Function to find neighbors within each batch
std::vector<std::pair<int, int>> find_neighbors_by_batch(const std::vector<std::vector<double>>& points, 
                                                         const std::vector<int>& batch_indices, 
                                                         double radius) {
    std::vector<std::pair<int, int>> neighbors;
    // Group points by batch
    std::unordered_map<int, std::vector<int>> batch_to_points;
    for (size_t i = 0; i < batch_indices.size(); ++i) {
        batch_to_points[batch_indices[i]].push_back(i);
    }

    // Process each batch independently
    for (const auto& batch : batch_to_points) {
        const auto& batch_points = batch.second;
        size_t num_points = batch_points.size();

        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = i + 1; j < num_points; ++j) {
                int idx1 = batch_points[i];
                int idx2 = batch_points[j];

                double dist = euclidean_distance(points[idx1], points[idx2]);
                if (dist <= radius) {
                    neighbors.emplace_back(idx1, idx2);
                    neighbors.emplace_back(idx2, idx1);  // Since the relation is symmetric
                }
            }
        }
    }
    return neighbors;
}

// Binding code
PYBIND11_MODULE(c_radius_graph, m) {
    m.def("find_neighbors_by_batch", &find_neighbors_by_batch, "Find neighbors within a given radius by batch",
          pybind11::arg("points"), pybind11::arg("batch_indices"), pybind11::arg("radius"));
}
