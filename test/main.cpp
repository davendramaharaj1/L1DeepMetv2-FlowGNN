
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "helper.h"

namespace py = pybind11;

void initialize_application() {
    load_weights();
    std::cout << "Weights loaded successfully.\n";
}

void run_graph_network(py::array_t<float> x_cont, py::array_t<int> x_cat) {
    // Convert py::array_t to native C++ arrays or std::vector
    auto buf_x_cont = x_cont.request();
    auto buf_x_cat = x_cat.request();

    if (buf_x_cont.ndim != 2 || buf_x_cat.ndim != 2) {
        throw std::runtime_error("Array dimensions mismatch.");
    }

    GraphMetNetworkLayer(
        reinterpret_cast<float (*)[CONT_DIM]>(buf_x_cont.ptr),
        reinterpret_cast<int (*)[CAT_DIM]>(buf_x_cat.ptr),
    );
}

PYBIND11_MODULE(appmodule, m) {
    m.def("initialize", &initialize_application, "Initialize the application");
    m.def("run_network", &run_graph_network, "Run the graph network");
}