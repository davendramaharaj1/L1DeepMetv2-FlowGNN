#ifndef __DCL_H__
#define __DCL_H__

/** Model Hyperparameters */
#define NUM_NODES 128   // this is a fixed value; receive an array of 128 particles every 54 cycles ~ 150ns
#define NUM_FEAT 8      // each particle has 8 features 
#define CONT_DIM 6      // 6 real number features
#define CAT_DIM 2       // 2 categorical features
#define HIDDEN_DIM 32   // size of hidden layers
#define OUTPUT_DIM 1

#define PDGS_SIZE 7
#define MAX_EDGES 1000
#define CONV_DEPTH 2

#define epsilon 0.00001
#define deltaR  0.4

/** self.pdgs */
extern int pdgs[PDGS_SIZE];

/** normalization */
extern float norm[CONT_DIM];

/** Learnable parameters for each layer in L1DeepMetv2 
 * 
 * These are mostly defined in the constructor for class GraphMetNetwork
*/
extern float graphmet_embed_charge_weight[3][8];
extern float graphmet_embed_pdgid_weight[7][8];
extern float graphmet_embed_continuous_0_weight[16][6];
extern float graphmet_embed_continuous_0_bias[16];
extern float graphmet_embed_categorical_0_weight[16][16];
extern float graphmet_embed_categorical_0_bias[16];
extern float graphmet_encode_all_weight[32][32];
extern float graphmet_encode_all_bias[32];
extern float graphmet_bn_all_weight[32];
extern float graphmet_bn_all_bias[32];
extern float graphmet_bn_all_running_mean[32];
extern float graphmet_bn_all_running_var[32];
extern float graphmet_bn_all_batches_tracked[1];
extern float graphmet_conv_continuous_0_0_nn_0_weight[32][64];
extern float graphmet_conv_continuous_0_0_nn_0_bias[32];
extern float graphmet_conv_continuous_0_1_weight[32];
extern float graphmet_conv_continuous_0_1_bias[32];
extern float graphmet_conv_continuous_0_1_running_mean[32];
extern float graphmet_conv_continuous_0_1_running_var[32];
extern float graphmet_conv_continuous_0_1_num_batches_tracked[1];
extern float graphmet_conv_continuous_1_0_nn_0_weight[32][64];
extern float graphmet_conv_continuous_1_0_nn_0_bias[32];
extern float graphmet_conv_continuous_1_1_weight[32];
extern float graphmet_conv_continuous_1_1_bias[32];
extern float graphmet_conv_continuous_1_1_running_mean[32];
extern float graphmet_conv_continuous_1_1_running_var[32];
extern float graphmet_conv_continuous_1_1_num_batches_tracked[1];
extern float graphmet_output_0_weight[16][32];
extern float graphmet_output_0_bias[16];
extern float graphmet_output_2_weight[1][16];
extern float graphmet_output_2_bias[1];


/* Function Prototypes for GNN inference */
void GraphMetNetworkLayer(float x_cont[NUM_NODES][CONT_DIM], int x_cat[NUM_NODES][CAT_DIM], int batch[NUM_NODES]);
void load_weights();



#endif