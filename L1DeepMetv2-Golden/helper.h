#ifndef __HELPER_H__
#define __HELPER_H__

#include "dcl.h"

/* Function Prototypes for GNN inference */
void GraphMetNetworkLayer(float x_cont[NUM_NODES][CONT_DIM], int x_cat[NUM_NODES][CAT_DIM], int batch[NUM_NODES]);

void load_weights();

/** implementation of torch.nn.ELU() 
 * 
 * @param x (float): value
 * @param alpha (float): value for ELU formulation (Default = 1.0)
*/
float ELU(float x, float alpha);

// Calculate squared distance
float squared_distance(float x1, float y1, float x2, float y2);


// Naive radius graph function
void radius_graph(float etaphi[][2], int num_nodes, float r, int batch[], int edge_index[][2], int *edge_cnt, int include_loop, int max_edges);

// Utility function to perform matrix multiplication followed by bias addition per individual data vector
void matmul_and_add_bias(float result[HIDDEN_DIM], float vector[HIDDEN_DIM * 2], float weight[HIDDEN_DIM][HIDDEN_DIM * 2], float bias[HIDDEN_DIM]);

void batch_normalization(float node_features[NUM_NODES][HIDDEN_DIM], 
                         float batch_weight[HIDDEN_DIM], 
                         float batch_bias[HIDDEN_DIM], 
                         float mean[HIDDEN_DIM], 
                         float variance[HIDDEN_DIM]);

// The edge convolution function with batch normalization and residual connection
void edge_convolution(int num_edges,
                      float emb[NUM_NODES][HIDDEN_DIM], 
                      int edge_index[][2], 
                      float weight[HIDDEN_DIM][HIDDEN_DIM * 2], 
                      float bias[HIDDEN_DIM],
                      float batch_weight[HIDDEN_DIM], 
                      float batch_bias[HIDDEN_DIM], 
                      float mean[HIDDEN_DIM], 
                      float variance[HIDDEN_DIM]);

// Utility function to perform matrix-vector multiplication and add bias
void matvec_multiply_and_add_bias(float result[HIDDEN_DIM/2], float mat[HIDDEN_DIM/2][HIDDEN_DIM], 
                                  float vec[HIDDEN_DIM], float bias[HIDDEN_DIM/2]);

// ELU activation function
void apply_elu(float arr[HIDDEN_DIM/2], float alpha);

// relu activation
void apply_relu(float arr[], int length);

// Output layer forward pass function
void forward_output_layer(float emb[NUM_NODES][HIDDEN_DIM], 
                          float weight1[HIDDEN_DIM/2][HIDDEN_DIM], float bias1[HIDDEN_DIM/2],
                          float weight2[OUTPUT_DIM][HIDDEN_DIM/2], float bias2[OUTPUT_DIM],
                          float output[NUM_NODES]);


#endif