#include "dcl.h"

// Calculate squared distance
float squared_distance(float x1, float y1, float x2, float y2)
{
    float dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    return dist;
}


// Define MLP (Multi-Layer Perceptron) function
void mlp(float input[], float weight1[], float bias1[], float weight2[], float bias2[], float output[]) {
    // Compute first linear layer
    float linear_output1[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) {
        linear_output1[i] = bias1[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            linear_output1[i] += input[j] * weight1[i * HIDDEN_DIM + j];
        }
    }
    
    // Apply ReLU activation
    for (int i = 0; i < HIDDEN_DIM; i++) {
        linear_output1[i] = (linear_output1[i] > 0) ? linear_output1[i] : 0;
    }

    // Compute second linear layer
    for (int i = 0; i < HIDDEN_DIM; i++) {
        output[i] = bias2[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            output[i] += linear_output1[j] * weight2[i * HIDDEN_DIM + j];
        }
    }
}

// Define edge convolution function using the fifth choice of h function with max pooling aggregation
void edge_conv(int num_nodes, int hidden_dim, int num_edges, float emb[][HIDDEN_DIM], int edge_src[], int edge_dst[], float weight1[], float bias1[], float weight2[], float bias2[], float output[][HIDDEN_DIM]) {
    // Initialize output with a large negative value
    float max_output[hidden_dim];
    for (int i = 0; i < hidden_dim; i++) {
        max_output[i] = -INFINITY;
    }

    // Loop through each edge
    for (int e = 0; e < num_edges; e++) {
        // Get source and destination indices for the edge
        int src_idx = edge_src[e];
        int dst_idx = edge_dst[e];
        
        // Compute the difference vector x_j - x_i
        float diff_vector[hidden_dim];
        for (int i = 0; i < hidden_dim; i++) {
            diff_vector[i] = emb[dst_idx][i] - emb[src_idx][i];
        }
        
        // Concatenate the difference vector with x_i
        float concat_features[2 * hidden_dim];
        for (int i = 0; i < hidden_dim; i++) {
            concat_features[i] = emb[src_idx][i];
            concat_features[i + hidden_dim] = diff_vector[i];
        }
        
        // Apply MLP to the concatenated features
        float mlp_output[hidden_dim];
        mlp(concat_features, weight1, bias1, weight2, bias2, mlp_output);
        
        // Update max_output using max pooling
        for (int i = 0; i < hidden_dim; i++) {
            if (mlp_output[i] > max_output[i]) {
                max_output[i] = mlp_output[i];
            }
        }

            // Assign the max_output to the output node
        for (int i = 0; i < hidden_dim; i++) {
            output[dst_idx][i] = max_output[i];
        }
    }
}



// Brute-force radius graph search
/**
 * Assume etaphi is flattened for storing 2D points
 * Must have batch array (data.batch in Python) for graph membership
 * 
 * @param out int edge_src []   src nodes
 * @param out int edge_dst []   dst nodes
*/
void radius_graph(float etaphi[], int batch[], float r, int edge_src[], int edge_dst[], int *num_edges) 
{
    int i, j;
    float r_squared = r * r;
    *num_edges = 0;
    
    for (i = 0; i < NUM_NODES; ++i) {
        for (j = 0; j < NUM_FEAT; ++j) {
            if (batch[i] == batch[j]) { // Check if i and j belong to the same graph
                float dist_sq = squared_distance(etaphi[2*i], etaphi[2*i + 1], etaphi[2*j], etaphi[2*j + 1]);
                if (dist_sq <= r_squared && i != j) { // Within radius and not the same node
                    edge_src[*num_edges] = i;
                    edge_dst[*num_edges] = j;
                    (*num_edges)++;
                    if (*num_edges == MAX_EDGES) return; // Prevent overflow of edge arrays
                }
            }
        }
    }
}

/** implementation of torch.nn.ELU() 
 * 
 * @param x (float): value
 * @param alpha (float): value for ELU formulation (Default = 1.0)
*/
float ELU(float x, float alpha)
{
    return x > 0 ? x : alpha*(exp(x)-1);
}

void Linear(float** input, float** output, float** weight, float* bias)
{
    for (int i = 0; i < NUM_NODES; i++)
    {
        for(int j = 0; j < HIDDEN_DIM/2; j++)
        {
            for(int k = 0; k < CONT_DIM; k++)
            {
                output[i][j] += input[i][k] * weight[j][k];
            }

            /** add the bias */
            output[i][j] += bias[j];

            /** apply torch.nn.ELU() to each element */
            output[i][j] = ELU(input[i][j], 1.0);
        }
    }
}

void BatchNorm1D(float** input, float** output, float* mean, float* var, float* weight, float* bias){
    for (int row = 0; row < NUM_NODES; row++)
    {
        for (int col = 0; col < HIDDEN_DIM; col++)
        {
            output[row][col] = ((input[row][col] - mean[col]) / (var[col] + epsilon))*weight[col]
                                + bias[col];
        }
    }
}

void EdgeConv2d(float** input, float** output, float** linear_weight, float* linear_bias, float* mean, float* var, float* batchNorm_weight, float* batchNorm_bias){
    float** temp;
    Linear(input, temp, linear_weight, linear_bias);
    BatchNorm1D(temp, output, mean, var, batchNorm_weight, batchNorm_bias);
}