#include "dcl.h"

/** intermediate variables in the forward function of GraphMetNetwork */
float emb_cont[NUM_NODES][HIDDEN_DIM/2];


/** implementation of torch.nn.ELU() 
 * 
 * @param x (float): value
 * @param alpha (float): value for ELU formulation (Default = 1.0)
*/
float ELU(float x, float alpha)
{
    return x > 0 ? x : alpha*(exp(x)-1);
}

void GraphMetNetworkLayer(float x_cont[NUM_NODES][CONT_DIM])
{

    /** emb_cont = self.embed_continuous(x_cont) */
    memset(emb_cont, 0, NUM_NODES * HIDDEN_DIM/2 * sizeof(float));
    /** shape: (NUM_NODES, CONT_DIM) * (CONT_DIM, HIDDEN_NUM//2)  = (NUM_NODES, HIDDEN_NUM//2) */
    for(int i = 0; i < NUM_NODES - 1; i++)
    {
        for(int j = 0; j < HIDDEN_DIM/2; j++)
        {
            emb_cont[i][j] = 0.0;
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
}