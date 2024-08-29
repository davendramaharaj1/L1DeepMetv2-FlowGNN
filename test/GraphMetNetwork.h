#include "parameters.h"

class GraphMetNetwork {
    public:

        // Constructor
        GraphMetNetwork();

        // Network Layers
        void GraphMetNetworkLayers(float x_cont[MAX_NODES][CONT_DIM], int x_cat[MAX_NODES][CAT_DIM], int num_nodes);

        // get the number of nodes used
        int get_num_nodes() { return this->num_nodes; }

        // methods to get intermediate variables
        const float* get_output() const { return output; }
        const float* get_emb_cont() const { return &emb_cont[0][0]; }
        const float* get_emb_chrg() const { return &emb_chrg[0][0]; }
        const float* get_emb_pdg() const { return &emb_pdg[0][0]; }
        const float* get_emb_cat() const { return &emb_cat[0][0]; }
        const float* get_emb() const { return &emb[0][0]; }
        const float* get_emb1() const { return &emb1[0][0]; }
        const float* get_emb2() const { return &emb2[0][0]; }

    private:
    
        // Intermediate layer outputs
        float emb_cont[MAX_NODES][HIDDEN_DIM/2];
        float emb_chrg[MAX_NODES][HIDDEN_DIM/4];
        float emb_pdg[MAX_NODES][HIDDEN_DIM/4];
        float emb_cat[MAX_NODES][HIDDEN_DIM/2];
        float encode_all[MAX_NODES][HIDDEN_DIM];
        float emb[MAX_NODES][HIDDEN_DIM];
        float emb1[MAX_NODES][HIDDEN_DIM];
        float emb2[MAX_NODES][HIDDEN_DIM];
        float output[MAX_NODES];

        // Internal variables
        float etaphi[MAX_NODES][2];
        int edge_index[MAX_EDGES][2];
        int num_edges;
        int num_nodes;

        // helper methods
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
        void radius_graph(float etaphi[][2], int num_nodes, float r, int edge_index[][2], int *edge_cnt, int include_loop, int max_edges);

        // Utility function to perform matrix multiplication followed by bias addition per individual data vector
        void matmul_and_add_bias(float result[HIDDEN_DIM], float vector[HIDDEN_DIM * 2], float weight[HIDDEN_DIM][HIDDEN_DIM * 2], float bias[HIDDEN_DIM]);

        void batch_normalization(float node_features[MAX_NODES][HIDDEN_DIM], 
                                float batch_weight[HIDDEN_DIM], 
                                float batch_bias[HIDDEN_DIM], 
                                float mean[HIDDEN_DIM], 
                                float variance[HIDDEN_DIM]);

        // The edge convolution function with batch normalization and residual connection
        void edge_convolution(int num_edges,
                            float emb[MAX_NODES][HIDDEN_DIM],
                            float emb_out[MAX_NODES][HIDDEN_DIM], 
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
        void forward_output_layer(float emb[MAX_NODES][HIDDEN_DIM], 
                                float weight1[HIDDEN_DIM/2][HIDDEN_DIM], float bias1[HIDDEN_DIM/2],
                                float weight2[OUTPUT_DIM][HIDDEN_DIM/2], float bias2[OUTPUT_DIM],
                                float output[MAX_NODES]);

};