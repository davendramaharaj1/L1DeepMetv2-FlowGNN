#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <string>
#include <iostream>
#include <stdexcept>

#include "GraphMetNetwork.h"

GraphMetNetwork::GraphMetNetwork() {
    // Constructor does nothing
}

void GraphMetNetwork::load_weights()
{
    FILE* f;
    std::string weights = "../weights/";

    // Helper function to open files and check for errors
    auto safe_fopen = [](const std::string& file_path) -> FILE* {
        FILE* file = fopen(file_path.c_str(), "rb");
        if (!file) {
            throw std::runtime_error("Error: Unable to open file " + file_path);
        }
        return file;
    };

    // Helper function to safely read data from files
    auto safe_fread = [](void* ptr, size_t size, size_t count, FILE* file, const std::string& file_path) {
        size_t read_count = fread(ptr, size, count, file);
        if (read_count != count) {
            throw std::runtime_error("Error: Unable to read the correct amount of data from " + file_path);
        }
    };

    try {
        // Load each weight file with error handling
        f = safe_fopen(weights + "graphnet.embed_charge.weight.bin");
        safe_fread(graphmet_embed_charge_weight, sizeof(float), 24, f, weights + "graphnet.embed_charge.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.embed_pdgid.weight.bin");
        safe_fread(graphmet_embed_pdgid_weight, sizeof(float), 56, f, weights + "graphnet.embed_pdgid.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.embed_continuous.0.weight.bin");
        safe_fread(graphmet_embed_continuous_0_weight, sizeof(float), 96, f, weights + "graphnet.embed_continuous.0.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.embed_continuous.0.bias.bin");
        safe_fread(graphmet_embed_continuous_0_bias, sizeof(float), 16, f, weights + "graphnet.embed_continuous.0.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.embed_categorical.0.weight.bin");
        safe_fread(graphmet_embed_categorical_0_weight, sizeof(float), 256, f, weights + "graphnet.embed_categorical.0.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.embed_categorical.0.bias.bin");
        safe_fread(graphmet_embed_categorical_0_bias, sizeof(float), 16, f, weights + "graphnet.embed_categorical.0.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.encode_all.0.weight.bin");
        safe_fread(graphmet_encode_all_weight, sizeof(float), 1024, f, weights + "graphnet.encode_all.0.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.encode_all.0.bias.bin");
        safe_fread(graphmet_encode_all_bias, sizeof(float), 32, f, weights + "graphnet.encode_all.0.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.bn_all.weight.bin");
        safe_fread(graphmet_bn_all_weight, sizeof(float), 32, f, weights + "graphnet.bn_all.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.bn_all.bias.bin");
        safe_fread(graphmet_bn_all_bias, sizeof(float), 32, f, weights + "graphnet.bn_all.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.bn_all.running_mean.bin");
        safe_fread(graphmet_bn_all_running_mean, sizeof(float), 32, f, weights + "graphnet.bn_all.running_mean.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.bn_all.running_var.bin");
        safe_fread(graphmet_bn_all_running_var, sizeof(float), 32, f, weights + "graphnet.bn_all.running_var.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.bn_all.num_batches_tracked.bin");
        safe_fread(graphmet_bn_all_batches_tracked, sizeof(float), 1, f, weights + "graphnet.bn_all.num_batches_tracked.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.0.0.nn.0.weight.bin");
        safe_fread(graphmet_conv_continuous_0_0_nn_0_weight, sizeof(float), 2048, f, weights + "graphnet.conv_continuous.0.0.nn.0.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.0.0.nn.0.bias.bin");
        safe_fread(graphmet_conv_continuous_0_0_nn_0_bias, sizeof(float), 32, f, weights + "graphnet.conv_continuous.0.0.nn.0.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.0.1.weight.bin");
        safe_fread(graphmet_conv_continuous_0_1_weight, sizeof(float), 32, f, weights + "graphnet.conv_continuous.0.1.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.0.1.bias.bin");
        safe_fread(graphmet_conv_continuous_0_1_bias, sizeof(float), 32, f, weights + "graphnet.conv_continuous.0.1.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.0.1.running_mean.bin");
        safe_fread(graphmet_conv_continuous_0_1_running_mean, sizeof(float), 32, f, weights + "graphnet.conv_continuous.0.1.running_mean.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.0.1.running_var.bin");
        safe_fread(graphmet_conv_continuous_0_1_running_var, sizeof(float), 32, f, weights + "graphnet.conv_continuous.0.1.running_var.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.0.1.num_batches_tracked.bin");
        safe_fread(graphmet_conv_continuous_0_1_num_batches_tracked, sizeof(float), 1, f, weights + "graphnet.conv_continuous.0.1.num_batches_tracked.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.1.0.nn.0.weight.bin");
        safe_fread(graphmet_conv_continuous_1_0_nn_0_weight, sizeof(float), 2048, f, weights + "graphnet.conv_continuous.1.0.nn.0.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.1.0.nn.0.bias.bin");
        safe_fread(graphmet_conv_continuous_1_0_nn_0_bias, sizeof(float), 32, f, weights + "graphnet.conv_continuous.1.0.nn.0.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.1.1.weight.bin");
        safe_fread(graphmet_conv_continuous_1_1_weight, sizeof(float), 32, f, weights + "graphnet.conv_continuous.1.1.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.1.1.bias.bin");
        safe_fread(graphmet_conv_continuous_1_1_bias, sizeof(float), 32, f, weights + "graphnet.conv_continuous.1.1.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.1.1.running_mean.bin");
        safe_fread(graphmet_conv_continuous_1_1_running_mean, sizeof(float), 32, f, weights + "graphnet.conv_continuous.1.1.running_mean.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.1.1.running_var.bin");
        safe_fread(graphmet_conv_continuous_1_1_running_var, sizeof(float), 32, f, weights + "graphnet.conv_continuous.1.1.running_var.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.conv_continuous.1.1.num_batches_tracked.bin");
        safe_fread(graphmet_conv_continuous_1_1_num_batches_tracked, sizeof(float), 1, f, weights + "graphnet.conv_continuous.1.1.num_batches_tracked.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.output.0.weight.bin");
        safe_fread(graphmet_output_0_weight, sizeof(float), 512, f, weights + "graphnet.output.0.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.output.0.bias.bin");
        safe_fread(graphmet_output_0_bias, sizeof(float), 16, f, weights + "graphnet.output.0.bias.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.output.2.weight.bin");
        safe_fread(graphmet_output_2_weight, sizeof(float), 16, f, weights + "graphnet.output.2.weight.bin");
        fclose(f);

        f = safe_fopen(weights + "graphnet.output.2.bias.bin");
        safe_fread(graphmet_output_2_bias, sizeof(float), 1, f, weights + "graphnet.output.2.bias.bin");
        fclose(f);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw; // Re-throw the exception after logging it
    }
}



float GraphMetNetwork::ELU(float x, float alpha)
{
    return x > 0 ? x : alpha*(exp(x)-1);
}

float GraphMetNetwork::squared_distance(float x1, float y1, float x2, float y2)
{
    float dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    return dist;
}

void GraphMetNetwork::radius_graph(float etaphi[][2], int num_nodes, float r, int edge_index[][2], int *edge_cnt, int include_loop, int max_edges) {

    int num_edges = 0;

    float r_squared = r * r;

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            if (!include_loop && i == j) {
                continue;
            }

            float dist_sq = squared_distance(etaphi[i][0], etaphi[i][1], etaphi[j][0], etaphi[j][1]);

            if (dist_sq <= r_squared) {

                if (num_edges < max_edges) {
                    edge_index[num_edges][0] = i;
                    edge_index[num_edges][1] = j;
                    num_edges++;
                } else {
                    // Edge list is full
                    printf("Maximum number of edges reached.\n");
                    break;
                }
            }
        }
    }

    *edge_cnt = num_edges;
}

void GraphMetNetwork::matmul_and_add_bias(float result[HIDDEN_DIM], float vector[HIDDEN_DIM * 2], float weight[HIDDEN_DIM][HIDDEN_DIM * 2], float bias[HIDDEN_DIM]) {
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < HIDDEN_DIM * 2; ++j) {
            result[i] += vector[j] * weight[i][j];
        }
        result[i] += bias[i];
    }
}

void GraphMetNetwork::batch_normalization(float node_features[MAX_NODES][HIDDEN_DIM], 
                         float batch_weight[HIDDEN_DIM], 
                         float batch_bias[HIDDEN_DIM], 
                         float mean[HIDDEN_DIM], 
                         float variance[HIDDEN_DIM]) {

    for (int i = 0; i < MAX_NODES; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            // Subtract mean and divide by the square root of variance
            float normalized = (node_features[i][j] - mean[j]) / sqrt(variance[j] + 1e-5);

            // Scale and shift
            node_features[i][j] = normalized * batch_weight[j] + batch_bias[j];
        }
    }
}

void GraphMetNetwork::edge_convolution(int num_edges,
                      float emb[MAX_NODES][HIDDEN_DIM], // input embedding
                      float emb_out[MAX_NODES][HIDDEN_DIM], // output embedding
                      int edge_index[][2], 
                      float weight[HIDDEN_DIM][HIDDEN_DIM * 2], 
                      float bias[HIDDEN_DIM],
                      float batch_weight[HIDDEN_DIM], 
                      float batch_bias[HIDDEN_DIM], 
                      float mean[HIDDEN_DIM], 
                      float variance[HIDDEN_DIM]) {

    // Temporary array to store the results of edge convolution operation
    float edge_conv_results[MAX_NODES][HIDDEN_DIM];
    for (int i = 0; i < MAX_NODES; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            edge_conv_results[i][j] = -FLT_MAX;
        }
    }


    // Temporary vector to store concatenated features and their difference
    float concatenated_features[HIDDEN_DIM * 2];

    // Perform edge convolution
    for (int i = 0; i < num_edges; ++i) {
        int src_node = edge_index[i][0];
        int neighbor_node = edge_index[i][1];

        // Concatenate features and compute difference
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            concatenated_features[j] = emb[src_node][j];
            concatenated_features[j + HIDDEN_DIM] = emb[neighbor_node][j] - emb[src_node][j];
        }

        // Transform concatenated features
        float transformed_features[HIDDEN_DIM];
        matmul_and_add_bias(transformed_features, concatenated_features, weight, bias);

        // Perform max pooling aggregation
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            edge_conv_results[src_node][j] = fmax(edge_conv_results[src_node][j], transformed_features[j]);
        }
    }

    // Apply batch normalization on edge_conv_results
    batch_normalization(edge_conv_results, batch_weight, batch_bias, mean, variance);

    // Perform the residual connection
    for (int i = 0; i < MAX_NODES; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            emb_out[i][j] = emb[i][j] + edge_conv_results[i][j];
        }
    }
}

// Utility function to perform matrix-vector multiplication and add bias
void GraphMetNetwork::matvec_multiply_and_add_bias(float result[HIDDEN_DIM/2], float mat[HIDDEN_DIM/2][HIDDEN_DIM], 
                                  float vec[HIDDEN_DIM], float bias[HIDDEN_DIM/2]) {
    for (int i = 0; i < HIDDEN_DIM/2; ++i) {
        result[i] = bias[i]; // start with bias
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
}

void GraphMetNetwork::apply_elu(float arr[HIDDEN_DIM/2], float alpha) {
    for (int i = 0; i < HIDDEN_DIM/2; ++i) {
        arr[i] = (arr[i] > 0) ? arr[i] : (expf(arr[i]) - 1) * alpha;
    }
}

void GraphMetNetwork::apply_relu(float arr[], int length) {
    for (int i = 0; i < length; ++i) {
        arr[i] = fmax(0, arr[i]); // ReLU is max(0, x)
    }
}

void GraphMetNetwork::forward_output_layer(float emb[MAX_NODES][HIDDEN_DIM], 
                          float weight1[HIDDEN_DIM/2][HIDDEN_DIM], float bias1[HIDDEN_DIM/2],
                          float weight2[OUTPUT_DIM][HIDDEN_DIM/2], float bias2[OUTPUT_DIM],
                          float output[MAX_NODES]) {

    // Temporary array to store the intermediate features after first linear layer
    float hidden[MAX_NODES][HIDDEN_DIM/2];

    // Temporary array for output
    float temp_out[MAX_NODES][OUTPUT_DIM];

    // Apply the first linear layer
    for (int i = 0; i < this->num_nodes; ++i) {
        matvec_multiply_and_add_bias(hidden[i], weight1, emb[i], bias1);
    }

    // Apply ELU activation function
    for (int i = 0; i < this->num_nodes; ++i) {
        apply_elu(hidden[i], 1.0); // assuming alpha is 1.0 for ELU
    }

    // Apply the second linear layer to get the final output
    for (int i = 0; i < this->num_nodes; ++i) {
        temp_out[i][0] = bias2[0]; // Initialize with bias for the output layer
        for (int k = 0; k < HIDDEN_DIM/2; ++k) {
            temp_out[i][0] += weight2[0][k] * hidden[i][k];
        }
    }

    // squeeze the '1' dimension from temp_out
    for (int i = 0; i < this->num_nodes; i++)
    {
        output[i] = temp_out[i][0];
    }
}

// Implementatin of GNN Layers...Equivalent of the Forward Layer
void GraphMetNetwork::GraphMetNetworkLayers(float x_cont[MAX_NODES][CONT_DIM], int x_cat[MAX_NODES][CAT_DIM], int num_nodes)
{
    // Set the number of nodes
    this->num_nodes = num_nodes;

    for (int i = 0; i < this->num_nodes; i++) {
        etaphi[i][0] = x_cont[i][3];  // 4th column (index 3) of data
        etaphi[i][1] = x_cont[i][4];  // 5th column (index 4) of data
    }

    // radius_graph(etaphi, MAX_NODES, batch, deltaR, edge_index, &num_edges);
    radius_graph(etaphi, this->num_nodes, deltaR, edge_index, &num_edges, 0, MAX_EDGES);


    // x_cont *= self.datanorm
    for (int i = 0; i < this->num_nodes; i++) {
        for (int j = 0; j < CONT_DIM; j++) {
            x_cont[i][j] *= norm[j];
        }
    }

    /** emb_cont = self.embed_continuous(x_cont) */
    memset(emb_cont, 0, MAX_NODES * HIDDEN_DIM/2 * sizeof(float));
    /** shape: (MAX_NODES, CONT_DIM) * (CONT_DIM, HIDDEN_NUM//2)  = (MAX_NODES, HIDDEN_NUM//2) */
    for (int i = 0; i < this->num_nodes; i++)
    {
        for(int j = 0; j < HIDDEN_DIM/2; j++)
        {
            for(int k = 0; k < CONT_DIM; k++)
            {
                emb_cont[i][j] += x_cont[i][k] * graphmet_embed_continuous_0_weight[j][k];
            }

            /** add the bias */
            emb_cont[i][j] += graphmet_embed_continuous_0_bias[j];

            /** apply torch.nn.ELU() to each element */
            emb_cont[i][j] = ELU(emb_cont[i][j], 1.0);
        }
    }

    /** emb_chrg = self.embed_charge(x_cat[:, 1] + 1) */
    for (int i = 0; i < this->num_nodes; i++)
    {
        /** get the indx */
        int idx = x_cat[i][1] + 1;

        for (int j = 0; j < HIDDEN_DIM/4; j++)
        {
            emb_chrg[i][j] = graphmet_embed_charge_weight[idx][j];
        }
    }

    /* pdg_remap = torch.abs(x_cat[:, 0]) */
    int pdg_remap[this->num_nodes];
    for (int i = 0; i < this->num_nodes; i++)
    {
        pdg_remap[i] = abs(x_cat[i][0]);
    }

    /**
     *  for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
    */
    for (int i = 0; i < PDGS_SIZE; i++)
    {
        int pdgval = pdgs[i];
        for (int row = 0; row < this->num_nodes; row++)
        {
            if (pdg_remap[row] == pdgval)
            {
                pdg_remap[row] = i;
            }
        }
    }

    /* emb_pdg = self.embed_pdgid(pdg_remap) */
    for (int i = 0; i < this->num_nodes; i++)
    {
        /** get the indx */
        int idx = pdg_remap[i];

        for (int j = 0; j < HIDDEN_DIM/4; j++)
        {
            emb_pdg[i][j] = graphmet_embed_pdgid_weight[idx][j];
        }
    }

    /** emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg], dim=1)) 
     * 
     * Shape: (MAX_NODES, HIDDEN_DIM/2) * (2*HIDDEN_DIM/4, HIDDEN_DIM/2) = (MAX_NODES, HIDDEN_DIM/2)
    */
    memset(emb_cat, 0, MAX_NODES * HIDDEN_DIM/2 * sizeof(float));

    for (int i = 0; i < this->num_nodes; i++) {
        for (int j = 0; j < HIDDEN_DIM / 2; j++) {
            // emb_cat[i][j] = 0;
            for (int k = 0; k < HIDDEN_DIM / 2; k++) {
                emb_cat[i][j] += graphmet_embed_categorical_0_weight[j][k] * (emb_chrg[i][k % (HIDDEN_DIM / 4)] + emb_pdg[i][k % (HIDDEN_DIM / 4)]);
            }
            emb_cat[i][j] += graphmet_embed_categorical_0_bias[j];

            emb_cat[i][j] = ELU(emb_cat[i][j], 1.0);
        }
    }

    /** self.encode_all(torch.cat([emb_cat, emb_cont], dim=1)) 
     * Shape: (MAX_NODES, HIDDEN_DIM) * (HIDDEN_DIM, HIDDEN_DIM) = (MAX_NODES, HIDDEN_DIM)
    */
    memset(emb, 0, MAX_NODES*HIDDEN_DIM*sizeof(float));
    for (int i = 0; i < this->num_nodes; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            emb[i][j] = 0;
            for (int k = 0; k < HIDDEN_DIM; k++) {
                emb[i][j] += graphmet_encode_all_weight[j][k] * (emb_cat[i][k % (HIDDEN_DIM / 2)] + emb_cont[i][k % (HIDDEN_DIM / 2)]);
            }
            emb[i][j] += graphmet_encode_all_bias[j];
            emb[i][j] = ELU(emb[i][j], 1.0);
        }
    }

    /* emb = self.bn_all(self.encode_all(torch.cat([emb_cat, emb_cont], dim=1))) */
    // memset(emb, 0, MAX_NODES*HIDDEN_DIM*sizeof(float));
    for (int row = 0; row < this->num_nodes; row++)
    {
        for (int col = 0; col < HIDDEN_DIM; col++)
        {
            emb[row][col] = ((encode_all[row][col] - graphmet_bn_all_running_mean[col]) / (graphmet_bn_all_running_var[col] + epsilon))*graphmet_bn_all_weight[col]
                                + graphmet_bn_all_bias[col];
        }
    }

    // perform the edge convolutions for CONV_DEPTH times
    // emb = emb + co_conv[1](co_conv[0](emb, edge_index))
    for (int i = 0; i < CONV_DEPTH; i++)
    {
        if (i == 0)
        {
            edge_convolution(
                num_edges,
                emb,
                emb1,
                edge_index,
                graphmet_conv_continuous_0_0_nn_0_weight,
                graphmet_conv_continuous_0_0_nn_0_bias,
                graphmet_conv_continuous_0_1_weight,
                graphmet_conv_continuous_0_1_bias,
                graphmet_conv_continuous_0_1_running_mean,
                graphmet_conv_continuous_0_1_running_var
                );
        }
        else
        {
            edge_convolution(
                num_edges,
                emb1,
                emb2,
                edge_index,
                graphmet_conv_continuous_1_0_nn_0_weight,
                graphmet_conv_continuous_1_0_nn_0_bias,
                graphmet_conv_continuous_1_1_weight,
                graphmet_conv_continuous_1_1_bias,
                graphmet_conv_continuous_1_1_running_mean,
                graphmet_conv_continuous_1_1_running_var
                );
        }
    }

    // out = self.output(emb)
    memset(output, 0, MAX_NODES*OUTPUT_DIM*sizeof(float));

    // Call the output layer
    forward_output_layer(emb2, graphmet_output_0_weight, graphmet_output_0_bias, graphmet_output_2_weight, graphmet_output_2_bias, output);

    // Apply ReLU to the final output
    apply_relu(output, this->num_nodes);

    return;
}