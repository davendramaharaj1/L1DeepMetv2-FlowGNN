#include "helper.h"

/** intermediate variables in the forward function of GraphMetNetwork */
float emb_cont[NUM_NODES][HIDDEN_DIM/2];
float emb_chrg[NUM_NODES][HIDDEN_DIM/4];
float emb_pdg[NUM_NODES][HIDDEN_DIM/4];
float emb_cat[NUM_NODES][HIDDEN_DIM/2];
float encode_all[NUM_NODES][HIDDEN_DIM];
float emb[NUM_NODES][HIDDEN_DIM];
float output[NUM_NODES];

// int edge_src[MAX_EDGES];
// int edge_dst[MAX_EDGES];
// float flattened_etaphi[NUM_NODES * 2];

float etaphi[NUM_NODES][2];
int edge_index[MAX_EDGES][2];

int num_edges;

void GraphMetNetworkLayer(float x_cont[NUM_NODES][CONT_DIM], int x_cat[NUM_NODES][CAT_DIM], int batch[NUM_NODES])
{

    /** create a flattened etaphi array by concatenating 4th and 5th columns of x_cont */
    // int index = 0;
    // for (int i = 0; i < NUM_NODES; i++) {
    //     flattened_etaphi[index] = x_cont[i][3]; // 4th column, index 3
    //     index++;
    //     flattened_etaphi[index] = x_cont[i][4]; // 5th column, index 4
    //     index++;
    // }

    for (int i = 0; i < NUM_NODES; i++) {
        etaphi[i][0] = x_cont[i][3];  // 4th column (index 3) of data
        etaphi[i][1] = x_cont[i][4];  // 5th column (index 4) of data
    }

    /** Generate a edges to create a graph */
    // radius_graph(flattened_etaphi, batch, deltaR, edge_src, edge_dst, &num_edges);
    radius_graph(etaphi, NUM_NODES, batch, deltaR, edge_index, &num_edges);


    // x_cont *= self.datanorm
    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < CONT_DIM; j++) {
            x_cont[i][j] *= norm[j];
        }
    }

    /** emb_cont = self.embed_continuous(x_cont) */
    memset(emb_cont, 0, NUM_NODES * HIDDEN_DIM/2 * sizeof(float));
    /** shape: (NUM_NODES, CONT_DIM) * (CONT_DIM, HIDDEN_NUM//2)  = (NUM_NODES, HIDDEN_NUM//2) */
    for (int i = 0; i < NUM_NODES; i++)
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
    for (int i = 0; i < NUM_NODES; i++)
    {
        /** get the indx */
        int idx = x_cat[i][1] + 1;

        for (int j = 0; j < HIDDEN_DIM/4; j++)
        {
            emb_chrg[i][j] = graphmet_embed_charge_weight[idx][j];
        }
    }

    /* pdg_remap = torch.abs(x_cat[:, 0]) */
    int pdg_remap[NUM_NODES];
    for (int i = 0; i < NUM_NODES; i++)
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
        for (int row = 0; row < NUM_NODES; row++)
        {
            if (pdg_remap[row] == pdgval)
            {
                pdg_remap[row] = i;
            }
        }
    }

    /* emb_pdg = self.embed_pdgid(pdg_remap) */
    for (int i = 0; i < NUM_NODES; i++)
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
     * Shape: (NUM_NODES, HIDDEN_DIM/2) * (2*HIDDEN_DIM/4, HIDDEN_DIM/2) = (NUM_NODES, HIDDEN_DIM/2)
    */
    memset(emb_cat, 0, NUM_NODES * HIDDEN_DIM/2 * sizeof(float));
    // for (int i = 0; i < NUM_NODES; i++)
    // {
    //     for (int j = 0; j < HIDDEN_DIM/2; j++)
    //     {
    //         for (int k = 0; k < HIDDEN_DIM/2; k++)
    //         {
    //             if (k < HIDDEN_DIM/4)
    //             {
    //                 emb_cat[i][j] += emb_chrg[i][k] * graphmet_embed_categorical_0_weight[j][k];
    //             }
    //             else{
    //                 emb_cat[i][j] += emb_pdg[i][k - HIDDEN_DIM/4] * graphmet_embed_categorical_0_weight[j][k];
    //             }
    //         }

    //         /** add the categorical bias */
    //         emb_cat[i][j] += graphmet_embed_categorical_0_bias[j];

    //         /** apply torch.nn.ELU() to each element */
    //         emb_cat[i][j] = ELU(emb_cat[i][j], 1.0);
    //     }
    // }

    for (int i = 0; i < NUM_NODES; i++) {
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
     * Shape: (NUM_NODES, HIDDEN_DIM) * (HIDDEN_DIM, HIDDEN_DIM) = (NUM_NODES, HIDDEN_DIM)
    */
    memset(emb, 0, NUM_NODES*HIDDEN_DIM*sizeof(float));
    // for (int i = 0; i < NUM_NODES; i++)
    // {
    //     for (int j = 0; j < HIDDEN_DIM; j++)
    //     {
    //         for (int k = 0; k < HIDDEN_DIM; k++)
    //         {
    //             if (k < HIDDEN_DIM/2)
    //             {
    //                 encode_all[i][j] += emb_cat[i][k] * graphmet_encode_all_weight[j][k];
    //             }
    //             else
    //             {
    //                 encode_all[i][j] += emb_cont[i][k-HIDDEN_DIM/2] * graphmet_encode_all_weight[j][k];
    //             }
    //         }

    //         /** add the bias */
    //         encode_all[i][j] += graphmet_encode_all_bias[j];

    //         /** apply torch.nn.ELU() to each element */
    //         encode_all[i][j] = ELU(encode_all[i][j], 1.0);
    //     }
    // }

    // float emb[NUM_NODES][HIDDEN_DIM];
    for (int i = 0; i < NUM_NODES; i++) {
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
    // memset(emb, 0, NUM_NODES*HIDDEN_DIM*sizeof(float));
    for (int row = 0; row < NUM_NODES; row++)
    {
        for (int col = 0; col < HIDDEN_DIM; col++)
        {
            emb[row][col] = ((encode_all[row][col] - graphmet_bn_all_running_mean[col]) / (graphmet_bn_all_running_var[col] + epsilon))*graphmet_bn_all_weight[col]
                                + graphmet_bn_all_bias[col];
        }
    }

    /**
     * 
     * At this point, the embeddings for every node has been calculated and stored in 
     * 
     * emb[NUM_NODES][HIDDEN_DIM] ---> 128 nodes with 32 hidden features
    */

    // perform the edge convolutions for CONV_DEPTH times
    // emb = emb + co_conv[1](co_conv[0](emb, edge_index))
    for (int i = 0; i < CONV_DEPTH; i++)
    {
        if (i == 0)
        {
            edge_convolution(
                num_edges,
                emb,
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
                emb,
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
    memset(output, 0, NUM_NODES*OUTPUT_DIM*sizeof(float));

    // Call the output layer
    forward_output_layer(emb, graphmet_output_0_weight, graphmet_output_0_bias, graphmet_output_2_weight, graphmet_output_2_bias, output);

    // Apply ReLU to the final output
    apply_relu(output, NUM_NODES);

    // Now, output contains the final results after the ReLU activation
    // print the outputs to verify
    for (int i = 0; i < NUM_NODES; i++) {
        printf("Output %d: %f\n", i, output[i]);
    }

    return;
}