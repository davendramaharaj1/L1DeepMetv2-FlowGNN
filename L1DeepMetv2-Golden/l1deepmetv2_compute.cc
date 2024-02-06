#include "dcl.h"

/** intermediate variables in the forward function of GraphMetNetwork */
float emb_cont[NUM_NODES][HIDDEN_DIM/2];
float emb_chrg[NUM_NODES][HIDDEN_DIM/4];
float emb_pdg[NUM_NODES][HIDDEN_DIM/4];
float emb_cat[NUM_NODES][HIDDEN_DIM/2];
float encode_all[NUM_NODES][HIDDEN_DIM];

/** implementation of torch.nn.ELU() 
 * 
 * @param x (float): value
 * @param alpha (float): value for ELU formulation (Default = 1.0)
*/
float ELU(float x, float alpha)
{
    return x > 0 ? x : alpha*(exp(x)-1);
}

void GraphMetNetworkLayer(float x_cont[NUM_NODES][CONT_DIM], int x_cat[NUM_NODES][CAT_DIM])
{

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
    for (int i = 0; i < NUM_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_DIM/2; j++)
        {
            for (int k = 0; k < HIDDEN_DIM/2; k++)
            {
                if (k < HIDDEN_DIM/4)
                {
                    emb_cat[i][j] += emb_chrg[i][k] * graphmet_embed_categorical_0_weight[j][k];
                }
                else{
                    emb_cat[i][j] += emb_pdg[i][k - HIDDEN_DIM/4] * graphmet_embed_categorical_0_weight[j][k];
                }
            }

            /** add the categorical bias */
            emb_cat[i][j] += graphmet_embed_categorical_0_bias[j];

            /** apply torch.nn.ELU() to each element */
            emb_cat[i][j] = ELU(emb_cat[i][j], 1.0);
        }
    }

    /** self.encode_all(torch.cat([emb_cat, emb_cont], dim=1)) 
     * Shape: (NUM_NODES, HIDDEN_DIM) * (HIDDEN_DIM, HIDDEN_DIM) = (NUM_NODES, HIDDEN_DIM)
    */
    memset(encode_all, 0, NUM_NODES*HIDDEN_DIM*sizeof(float));
    for (int i = 0; i < NUM_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_DIM; j++)
        {
            for (int k = 0; k < HIDDEN_DIM; k++)
            {
                if (k < HIDDEN_DIM/2)
                {
                    encode_all[i][j] += emb_cat[i][k] * graphmet_encode_all_weight[j][k];
                }
                else
                {
                    encode_all[i][j] += emb_cont[i][k-HIDDEN_DIM/2] * graphmet_encode_all_weight[j][k];
                }
            }

            /** add the bias */
            encode_all[i][j] += graphmet_encode_all_bias[j];

            /** apply torch.nn.ELU() to each element */
            encode_all[i][j] = ELU(encode_all[i][j], 1.0);
        }
    }
}