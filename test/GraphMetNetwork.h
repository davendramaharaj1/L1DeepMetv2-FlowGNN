#include <vector>
#include <unordered_map>
#include "parameters.h"

class GraphMetNetwork {
    public:

        // Constructor
        GraphMetNetwork();

        // Network Layers
        void GraphMetNetworkLayers(float x_cont[MAX_NODES][CONT_DIM], int x_cat[MAX_NODES][CAT_DIM], int batch[MAX_NODES], int num_nodes);

        // helper methods
        void load_weights(std::string weights);

        // getters for inputs
        const float* get_x_cont() const { return &_x_cont[0][0]; }
        const int* get_x_cat() const { return &_x_cat[0][0]; }
        const int* get_batch() const { return _batch; }
        int get_num_nodes() { return this->_num_nodes; }
        
        // getters for internal variables
        int get_num_edges() { return this->num_edges; }
        int* get_edge_index() { return &edge_index[0][0]; }
        float* get_etaphi() { return &etaphi[0][0]; }

        // methods to get intermediate variables
        const float* get_output() const { return output; }
        const float* get_emb_cont() const { return &emb_cont[0][0]; }
        const float* get_emb_chrg() const { return &emb_chrg[0][0]; }
        const float* get_emb_pdg() const { return &emb_pdg[0][0]; }
        const float* get_emb_cat() const { return &emb_cat[0][0]; }
        const float* get_encode_all() const { return &encode_all[0][0]; }
        const float* get_emb() const { return &emb[0][0]; }
        const float* get_emb1() const { return &emb1[0][0]; }
        const float* get_emb2() const { return &emb2[0][0]; }
        const float* get_emb_0_0() const { return &emb_0_0[0][0]; }
        const float* get_emb_0_1() const { return &emb_0_1[0][0]; }
        const float* get_emb_1_0() const { return &emb_1_0[0][0]; }
        const float* get_emb_1_1() const { return &emb_1_1[0][0]; }

        // Getter methods for weights
        const float* get_graphnet_embed_charge_weight() const { return &graphnet_embed_charge_weight[0][0]; }
        const float* get_graphnet_embed_pdgid_weight() const { return &graphnet_embed_pdgid_weight[0][0]; }
        const float* get_graphnet_embed_continuous_0_weight() const { return &graphnet_embed_continuous_0_weight[0][0]; }
        const float* get_graphnet_embed_continuous_0_bias() const { return &graphnet_embed_continuous_0_bias[0]; }
        const float* get_graphnet_embed_categorical_0_weight() const { return &graphnet_embed_categorical_0_weight[0][0]; }
        const float* get_graphnet_embed_categorical_0_bias() const { return &graphnet_embed_categorical_0_bias[0]; }
        const float* get_graphnet_encode_all_0_weight() const { return &graphnet_encode_all_0_weight[0][0]; }
        const float* get_graphnet_encode_all_0_bias() const { return &graphnet_encode_all_0_bias[0]; }
        const float* get_graphnet_bn_all_weight() const { return &graphnet_bn_all_weight[0]; }
        const float* get_graphnet_bn_all_bias() const { return &graphnet_bn_all_bias[0]; }
        const float* get_graphnet_bn_all_running_mean() const { return &graphnet_bn_all_running_mean[0]; }
        const float* get_graphnet_bn_all_running_var() const { return &graphnet_bn_all_running_var[0]; }
        const int* get_graphnet_bn_all_num_batches_tracked() const { return &graphnet_bn_all_num_batches_tracked[0]; }
        const float* get_graphnet_conv_continuous_0_0_nn_0_weight() const { return &graphnet_conv_continuous_0_0_nn_0_weight[0][0]; }
        const float* get_graphnet_conv_continuous_0_0_nn_0_bias() const { return &graphnet_conv_continuous_0_0_nn_0_bias[0]; }
        const float* get_graphnet_conv_continuous_0_1_weight() const { return &graphnet_conv_continuous_0_1_weight[0]; }
        const float* get_graphnet_conv_continuous_0_1_bias() const { return &graphnet_conv_continuous_0_1_bias[0]; }
        const float* get_graphnet_conv_continuous_0_1_running_mean() const { return &graphnet_conv_continuous_0_1_running_mean[0]; }
        const float* get_graphnet_conv_continuous_0_1_running_var() const { return &graphnet_conv_continuous_0_1_running_var[0]; }
        const int* get_graphnet_conv_continuous_0_1_num_batches_tracked() const { return &graphnet_conv_continuous_0_1_num_batches_tracked[0]; }
        const float* get_graphnet_conv_continuous_1_0_nn_0_weight() const { return &graphnet_conv_continuous_1_0_nn_0_weight[0][0]; }
        const float* get_graphnet_conv_continuous_1_0_nn_0_bias() const { return &graphnet_conv_continuous_1_0_nn_0_bias[0]; }
        const float* get_graphnet_conv_continuous_1_1_weight() const { return &graphnet_conv_continuous_1_1_weight[0]; }
        const float* get_graphnet_conv_continuous_1_1_bias() const { return &graphnet_conv_continuous_1_1_bias[0]; }
        const float* get_graphnet_conv_continuous_1_1_running_mean() const { return &graphnet_conv_continuous_1_1_running_mean[0]; }
        const float* get_graphnet_conv_continuous_1_1_running_var() const { return &graphnet_conv_continuous_1_1_running_var[0]; }
        const int* get_graphnet_conv_continuous_1_1_num_batches_tracked() const { return &graphnet_conv_continuous_1_1_num_batches_tracked[0]; }
        const float* get_graphnet_output_0_weight() const { return &graphnet_output_0_weight[0][0]; }
        const float* get_graphnet_output_0_bias() const { return &graphnet_output_0_bias[0]; }
        const float* get_graphnet_output_2_weight() const { return &graphnet_output_2_weight[0][0]; }
        const float* get_graphnet_output_2_bias() const { return &graphnet_output_2_bias[0]; }


    private:

        /** self.pdgs */
        int pdgs[PDGS_SIZE] = {1, 2, 11, 13, 22, 130, 211};

        /** normalization */
        float norm[CONT_DIM] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        // Weight arrays
        float graphnet_embed_charge_weight[3][8];
        float graphnet_embed_pdgid_weight[7][8];
        float graphnet_embed_continuous_0_weight[16][6];
        float graphnet_embed_continuous_0_bias[16];
        float graphnet_embed_categorical_0_weight[16][16];
        float graphnet_embed_categorical_0_bias[16];
        float graphnet_encode_all_0_weight[32][32];
        float graphnet_encode_all_0_bias[32];
        float graphnet_bn_all_weight[32];
        float graphnet_bn_all_bias[32];
        float graphnet_bn_all_running_mean[32];
        float graphnet_bn_all_running_var[32];
        int graphnet_bn_all_num_batches_tracked[1];
        float graphnet_conv_continuous_0_0_nn_0_weight[32][64];
        float graphnet_conv_continuous_0_0_nn_0_bias[32];
        float graphnet_conv_continuous_0_1_weight[32];
        float graphnet_conv_continuous_0_1_bias[32];
        float graphnet_conv_continuous_0_1_running_mean[32];
        float graphnet_conv_continuous_0_1_running_var[32];
        int graphnet_conv_continuous_0_1_num_batches_tracked[1];
        float graphnet_conv_continuous_1_0_nn_0_weight[32][64];
        float graphnet_conv_continuous_1_0_nn_0_bias[32];
        float graphnet_conv_continuous_1_1_weight[32];
        float graphnet_conv_continuous_1_1_bias[32];
        float graphnet_conv_continuous_1_1_running_mean[32];
        float graphnet_conv_continuous_1_1_running_var[32];
        int graphnet_conv_continuous_1_1_num_batches_tracked[1];
        float graphnet_output_0_weight[16][32];
        float graphnet_output_0_bias[16];
        float graphnet_output_2_weight[1][16];
        float graphnet_output_2_bias[1];

        // Inputs
        float _x_cont[MAX_NODES][CONT_DIM];
        int _x_cat[MAX_NODES][CAT_DIM];
        int _batch[MAX_NODES];
        int _num_nodes;
    
        // Intermediate layer outputs
        float emb_cont[MAX_NODES][HIDDEN_DIM/2];
        float emb_chrg[MAX_NODES][HIDDEN_DIM/4];
        float emb_pdg[MAX_NODES][HIDDEN_DIM/4];
        float emb_cat[MAX_NODES][HIDDEN_DIM/2];
        float encode_all[MAX_NODES][HIDDEN_DIM];
        float emb[MAX_NODES][HIDDEN_DIM];
        float emb1[MAX_NODES][HIDDEN_DIM];
        float emb2[MAX_NODES][HIDDEN_DIM];
        float emb_0_0[MAX_NODES][HIDDEN_DIM];
        float emb_0_1[MAX_NODES][HIDDEN_DIM];
        float emb_1_0[MAX_NODES][HIDDEN_DIM];
        float emb_1_1[MAX_NODES][HIDDEN_DIM];
        float output[MAX_NODES];

        // Internal variables
        float etaphi[MAX_NODES][2];
        int edge_index[MAX_EDGES][2];
        int num_edges;

        void reset();

        /** implementation of torch.nn.ELU() 
         * 
         * @param x (float): value
         * @param alpha (float): value for ELU formulation (Default = 1.0)
        */
        float ELU(float x, float alpha);

        float euclidean(float point1[2], float point2[2]);

        // Function to calculate the Euclidean distance between two points
        float euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2);

        std::vector<std::pair<int, int>> find_neighbors_by_batch(const std::vector<std::vector<double>>& points,
                                                            double radius, 
                                                            const std::vector<int>& batch_indices
                                                            );

        // Calculate squared distance
        float squared_distance(float x1, float y1, float x2, float y2);


        // Naive radius graph function
        // void radius_graph(float etaphi[][2], int batch[MAX_NODES], int num_nodes, float r, int edge_index[][2], int *edge_cnt, int include_loop, int max_edges);
        void radius_graph(float etaphi[][2], int batch[MAX_NODES], int num_nodes, float r, int max_edges);


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

        // sigmoid 
        void sigmoid(float array[], uint64_t size);

        // Output layer forward pass function
        void forward_output_layer(float emb[MAX_NODES][HIDDEN_DIM], 
                                float weight1[HIDDEN_DIM/2][HIDDEN_DIM], float bias1[HIDDEN_DIM/2],
                                float weight2[OUTPUT_DIM][HIDDEN_DIM/2], float bias2[OUTPUT_DIM],
                                float output[MAX_NODES]);

};