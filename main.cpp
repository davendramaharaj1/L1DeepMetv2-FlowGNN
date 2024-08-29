
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "helper.h"

namespace py = pybind11;

void initialize_application() {
    load_weights();
    std::cout << "Weights loaded successfully.\n";
}

void run_graph_network(py::array_t<float> x_cont, py::array_t<int> x_cat, py::array_t<int> batch) {
    // Convert py::array_t to native C++ arrays or std::vector
    auto buf_x_cont = x_cont.request();
    auto buf_x_cat = x_cat.request();
    auto buf_batch = batch.request();

    if (buf_x_cont.ndim != 2 || buf_x_cat.ndim != 2 || buf_batch.ndim != 1) {
        throw std::runtime_error("Array dimensions mismatch.");
    }

    GraphMetNetworkLayer(
        reinterpret_cast<float (*)[CONT_DIM]>(buf_x_cont.ptr),
        reinterpret_cast<int (*)[CAT_DIM]>(buf_x_cat.ptr),
        reinterpret_cast<int *>(buf_batch.ptr)
    );
}

PYBIND11_MODULE(appmodule, m) {
    m.def("initialize", &initialize_application, "Initialize the application");
    m.def("run_network", &run_graph_network, "Run the graph network");
}

// int main() {
//     // In the C++ domain, this would now be a waiting function or just end.
//     std::cout << "Module loaded and ready for Python calls.\n";
//     return 0; // Keep this if the main function needs to do nothing else.
// }

// int main()
// {

//     printf("\n******* Beginning GNN inference using L1DeepMetv2 *******\n");

//     printf("\n******* Loading Weights *******\n");

//     load_weights();

//     printf("\n******* Done Loading Weights *******\n");


//     GraphMetNetworkLayer(x_cont[NUM_NODES][CONT_DIM], x_cat[NUM_NODES][CAT_DIM], batch[NUM_NODES]);

//     return 0;
// }