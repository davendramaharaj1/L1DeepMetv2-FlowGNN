#include "dcl.h"

/** implementation of torch.nn.ELU() 
 * 
 * @param x (float): value
 * @param alpha (float): value for ELU formulation (Default = 1.0)
*/
float ELU(float x, float alpha)
{
    return x > 0 ? x : alpha*(exp(x)-1);
}

// Calculate squared distance
float squared_distance(float x1, float y1, float x2, float y2)
{
    float dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    return dist;
}

void radius_graph(float etaphi[][2], int numNodes, int batch[], float delta, int edge_index[][2], int *edge_index_size) {
    int edge_count = 0;

    for (int i = 0; i < numNodes; i++) {
        for (int j = i + 1; j < numNodes; j++) {
            float dx = etaphi[i][0] - etaphi[j][0];
            float dy = etaphi[i][1] - etaphi[j][1];
            float distance = sqrt(dx * dx + dy * dy);

            if (distance <= delta && i != j) {
                edge_index[edge_count][0] = i;
                edge_index[edge_count][1] = j;
                edge_count++;
            }
        }
    }

    *edge_index_size = edge_count;
}

// Utility function to perform matrix multiplication followed by bias addition per individual data vector
void matmul_and_add_bias(float result[HIDDEN_DIM], float vector[HIDDEN_DIM * 2], float weight[HIDDEN_DIM][HIDDEN_DIM * 2], float bias[HIDDEN_DIM]) {
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < HIDDEN_DIM * 2; ++j) {
            result[i] += vector[j] * weight[i][j];
        }
        result[i] += bias[i];
    }
}

void batch_normalization(float node_features[NUM_NODES][HIDDEN_DIM], 
                         float batch_weight[HIDDEN_DIM], 
                         float batch_bias[HIDDEN_DIM], 
                         float mean[HIDDEN_DIM], 
                         float variance[HIDDEN_DIM]) {

    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            // Subtract mean and divide by the square root of variance
            float normalized = (node_features[i][j] - mean[j]) / sqrtf(variance[j] + 1e-5);

            // Scale and shift
            node_features[i][j] = normalized * batch_weight[j] + batch_bias[j];
        }
    }
}

// The edge convolution function with batch normalization and residual connection
void edge_convolution(int num_edges,
                      float emb[NUM_NODES][HIDDEN_DIM], 
                      int edge_index[][2], 
                      float weight[HIDDEN_DIM][HIDDEN_DIM * 2], 
                      float bias[HIDDEN_DIM],
                      float batch_weight[HIDDEN_DIM], 
                      float batch_bias[HIDDEN_DIM], 
                      float mean[HIDDEN_DIM], 
                      float variance[HIDDEN_DIM]) {

    // Temporary array to store the results of edge convolution operation
    float edge_conv_results[NUM_NODES][HIDDEN_DIM] = {0};

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
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            emb[i][j] += edge_conv_results[i][j];
        }
    }
}

// Utility function to perform matrix-vector multiplication and add bias
void matvec_multiply_and_add_bias(float result[HIDDEN_DIM/2], float mat[HIDDEN_DIM/2][HIDDEN_DIM], 
                                  float vec[HIDDEN_DIM], float bias[HIDDEN_DIM/2]) {
    for (int i = 0; i < HIDDEN_DIM/2; ++i) {
        result[i] = bias[i]; // start with bias
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
}

// ELU activation function
void apply_elu(float arr[HIDDEN_DIM/2], float alpha) {
    for (int i = 0; i < HIDDEN_DIM/2; ++i) {
        arr[i] = (arr[i] > 0) ? arr[i] : (expf(arr[i]) - 1) * alpha;
    }
}

// relu activation
void apply_relu(float arr[], int length) {
    for (int i = 0; i < length; ++i) {
        arr[i] = fmax(0, arr[i]); // ReLU is max(0, x)
    }
}

// Output layer forward pass function
void forward_output_layer(float emb[NUM_NODES][HIDDEN_DIM], 
                          float weight1[HIDDEN_DIM/2][HIDDEN_DIM], float bias1[HIDDEN_DIM/2],
                          float weight2[OUTPUT_DIM][HIDDEN_DIM/2], float bias2[OUTPUT_DIM],
                          float output[NUM_NODES]) {

    // Temporary array to store the intermediate features after first linear layer
    float hidden[NUM_NODES][HIDDEN_DIM/2];

    // Temporary array for output
    float temp_out[NUM_NODES][OUTPUT_DIM];

    // Apply the first linear layer
    for (int i = 0; i < NUM_NODES; ++i) {
        matvec_multiply_and_add_bias(hidden[i], weight1, emb[i], bias1);
    }

    // Apply ELU activation function
    for (int i = 0; i < NUM_NODES; ++i) {
        apply_elu(hidden[i], 1.0); // assuming alpha is 1.0 for ELU
    }

    // Apply the second linear layer to get the final output
    for (int i = 0; i < NUM_NODES; ++i) {
        temp_out[i][0] = bias2[0]; // Initialize with bias for the output layer
        for (int k = 0; k < HIDDEN_DIM/2; ++k) {
            temp_out[i][0] += weight2[0][k] * hidden[i][k];
        }
    }

    // squeeze the '1' dimension from temp_out
    for (int i = 0; i < NUM_NODES; i++)
    {
        output[i] = temp_out[i][0];
    }
}




// // Define MLP (Multi-Layer Perceptron) function
// void mlp(float input[], float weight1[][HIDDEN_DIM*2], float bias1[], float output[]) {
//     // Compute first linear layer
//     float linear_output1[HIDDEN_DIM];
//     for (int i = 0; i < HIDDEN_DIM; i++) {
//         linear_output1[i] = bias1[i];
//         for (int j = 0; j < HIDDEN_DIM; j++) {
//             linear_output1[i] += input[j] * weight1[i * HIDDEN_DIM + j];
//         }
//     }
    
//     // Apply ReLU activation
//     for (int i = 0; i < HIDDEN_DIM; i++) {
//         linear_output1[i] = (linear_output1[i] > 0) ? linear_output1[i] : 0;
//     }
// }

// Define edge convolution function using the fifth choice of h function with max pooling aggregation
// void edge_conv(int num_nodes, int hidden_dim, int num_edges, float emb[][HIDDEN_DIM], int edge_idx[][2], float weight1[][2*HIDDEN_DIM], float bias1[], float output[][HIDDEN_DIM]) {
//     // Initialize output with a large negative value
//     float max_output[hidden_dim];
//     for (int i = 0; i < hidden_dim; i++) {
//         max_output[i] = -1000;
//     }

//     // Loop through each edge
//     for (int e = 0; e < num_edges; e++) {
//         // Get source and destination indices for the edge
//         int src_idx = edge_idx[e][0];
//         int dst_idx = edge_idx[e][1];
        
//         // Compute the difference vector x_j - x_i
//         float diff_vector[hidden_dim];
//         for (int i = 0; i < hidden_dim; i++) {
//             diff_vector[i] = emb[dst_idx][i] - emb[src_idx][i];
//         }
        
//         // Concatenate the difference vector with x_i
//         float concat_features[2 * hidden_dim];
//         for (int i = 0; i < hidden_dim; i++) {
//             concat_features[i] = emb[src_idx][i];
//             concat_features[i + hidden_dim] = diff_vector[i];
//         }
        
//         // Apply MLP to the concatenated features
//         float mlp_output[hidden_dim];
//         mlp(concat_features, weight1, bias1, mlp_output);
        
//         // Update max_output using max pooling
//         for (int i = 0; i < hidden_dim; i++) {
//             if (mlp_output[i] > max_output[i]) {
//                 max_output[i] = mlp_output[i];
//             }
//         }

//             // Assign the max_output to the output node
//         for (int i = 0; i < hidden_dim; i++) {
//             output[dst_idx][i] = max_output[i];
//         }
//     }
// }

// void Linear(float** input, float** output, float** weight, float* bias)
// {
//     for (int i = 0; i < NUM_NODES; i++)
//     {
//         for(int j = 0; j < HIDDEN_DIM/2; j++)
//         {
//             for(int k = 0; k < CONT_DIM; k++)
//             {
//                 output[i][j] += input[i][k] * weight[j][k];
//             }

//             /** add the bias */
//             output[i][j] += bias[j];

//             /** apply torch.nn.ELU() to each element */
//             output[i][j] = ELU(input[i][j], 1.0);
//         }
//     }
// }

// void BatchNorm1D(float** input, float** output, float* mean, float* var, float* weight, float* bias){
//     for (int row = 0; row < NUM_NODES; row++)
//     {
//         for (int col = 0; col < HIDDEN_DIM; col++)
//         {
//             output[row][col] = ((input[row][col] - mean[col]) / (var[col] + epsilon))*weight[col]
//                                 + bias[col];
//         }
//     }
// }

// Brute-force radius graph search
/**
 * Assume etaphi is flattened for storing 2D points
 * Must have batch array (data.batch in Python) for graph membership
 * 
 * @param out int edge_src []   src nodes
 * @param out int edge_dst []   dst nodes
*/
// void radius_graph(float etaphi[], int batch[], float r, int edge_src[], int edge_dst[], int *num_edges) 
// {
//     int i, j;
//     float r_squared = r * r;
//     *num_edges = 0;
    
//     for (i = 0; i < NUM_NODES; ++i) {
//         for (j = 0; j < NUM_FEAT; ++j) {
//             if (batch[i] == batch[j]) { // Check if i and j belong to the same graph
//                 float dist_sq = squared_distance(etaphi[2*i], etaphi[2*i + 1], etaphi[2*j], etaphi[2*j + 1]);
//                 if (dist_sq <= r_squared && i != j) { // Within radius and not the same node
//                     edge_src[*num_edges] = i;
//                     edge_dst[*num_edges] = j;
//                     (*num_edges)++;
//                     if (*num_edges == MAX_EDGES) return; // Prevent overflow of edge arrays
//                 }
//             }
//         }
//     }
// }