#include <stdlib.h>
#include <stdio.h>
#include "dcl.h"

/** self.pdgs */
int pdgs[PDGS_SIZE] = {1, 2, 11, 13, 22, 130, 211};

/** normalization */
float norm[CONT_DIM] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

float graphmet_embed_charge_weight[3][8];
float graphmet_embed_pdgid_weight[7][8];
float graphmet_embed_continuous_0_weight[16][6];
float graphmet_embed_continuous_0_bias[16];
float graphmet_embed_categorical_0_weight[16][16];
float graphmet_embed_categorical_0_bias[16];
float graphmet_encode_all_weight[32][32];
float graphmet_encode_all_bias[32];
float graphmet_bn_all_weight[32];
float graphmet_bn_all_bias[32];
float graphmet_bn_all_running_mean[32];
float graphmet_bn_all_running_var[32];
float graphmet_bn_all_batches_tracked[1];
float graphmet_conv_continuous_0_0_nn_0_weight[32][64];
float graphmet_conv_continuous_0_0_nn_0_bias[32];
float graphmet_conv_continuous_0_1_weight[32];
float graphmet_conv_continuous_0_1_bias[32];
float graphmet_conv_continuous_0_1_running_mean[32];
float graphmet_conv_continuous_0_1_running_var[32];
float graphmet_conv_continuous_0_1_num_batches_tracked[1];
float graphmet_conv_continuous_1_0_nn_0_weight[32][64];
float graphmet_conv_continuous_1_0_nn_0_bias[32];
float graphmet_conv_continuous_1_1_weight[32];
float graphmet_conv_continuous_1_1_bias[32];
float graphmet_conv_continuous_1_1_running_mean[32];
float graphmet_conv_continuous_1_1_running_var[32];
float graphmet_conv_continuous_1_1_num_batches_tracked[1];
float graphmet_output_0_weight[16][32];
float graphmet_output_0_bias[16];
float graphmet_output_2_weight[1][16];
float graphmet_output_2_bias[1];


void load_weights()
{
    FILE* f;

    f = fopen("graphnet.embed_charge.weight.bin", "r");
    fread(graphmet_embed_charge_weight, sizeof(float), 24, f);
    fclose(f);

    f = fopen("graphnet.embed_pdgid.weight.bin", "r");
    fread(graphmet_embed_pdgid_weight, sizeof(float), 56, f);
    fclose(f);

    f = fopen("graphnet.embed_continuous.0.weight.bin", "r");
    fread(graphmet_embed_continuous_0_weight, sizeof(float), 96, f);
    fclose(f);

    f = fopen("graphnet.embed_continuous.0.bias.bin", "r");
    fread(graphmet_embed_continuous_0_bias, sizeof(float), 16, f);
    fclose(f);

    f = fopen("graphnet.embed_categorical.0.weight.bin", "r");
    fread(graphmet_embed_categorical_0_weight, sizeof(float), 256, f);
    fclose(f);

    f = fopen("graphnet.embed_categorical.0.bias.bin", "r");
    fread(graphmet_embed_categorical_0_bias, sizeof(float), 16, f);
    fclose(f);

    f = fopen("graphnet.encode_all.0.weight.bin", "r");
    fread(graphmet_encode_all_weight, sizeof(float), 1024, f);
    fclose(f);

    f = fopen("graphnet.encode_all.0.bias.bin", "r");
    fread(graphmet_encode_all_bias, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.bn_all.weight.bin", "r");
    fread(graphmet_bn_all_weight, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.bn_all.bias.bin", "r");
    fread(graphmet_bn_all_bias, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.bn_all.running_mean.bin", "r");
    fread(graphmet_bn_all_running_mean, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.bn_all.running_var.bin", "r");
    fread(graphmet_bn_all_running_var, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.bn_all.num_batches_tracked.bin", "r");
    fread(graphmet_bn_all_batches_tracked, sizeof(float), 1, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.0.0.nn.0.weight.bin", "r");
    fread(graphmet_conv_continuous_0_0_nn_0_weight, sizeof(float), 2048, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.0.0.nn.0.bias.bin", "r");
    fread(graphmet_conv_continuous_0_0_nn_0_bias, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.0.1.weight.bin", "r");
    fread(graphmet_conv_continuous_0_1_weight, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.0.1.bias.bin", "r");
    fread(graphmet_conv_continuous_0_1_bias, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.0.1.running_mean.bin", "r");
    fread(graphmet_conv_continuous_0_1_running_mean, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.0.1.running_var.bin", "r");
    fread(graphmet_conv_continuous_0_1_running_var, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.0.1.num_batches_tracked.bin", "r");
    fread(graphmet_conv_continuous_0_1_num_batches_tracked, sizeof(float), 1, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.1.0.nn.0.weight.bin", "r");
    fread(graphmet_conv_continuous_1_0_nn_0_weight, sizeof(float), 2048, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.1.0.nn.0.bias.bin", "r");
    fread(graphmet_conv_continuous_1_0_nn_0_bias, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.1.1.weight.bin", "r");
    fread(graphmet_conv_continuous_1_1_weight, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.1.1.bias.bin", "r");
    fread(graphmet_conv_continuous_1_1_bias, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.1.1.running_mean.bin", "r");
    fread(graphmet_conv_continuous_1_1_running_mean, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.1.1.running_var.bin", "r");
    fread(graphmet_conv_continuous_1_1_running_var, sizeof(float), 32, f);
    fclose(f);

    f = fopen("graphnet.conv_continuous.1.1.num_batches_tracked.bin", "r");
    fread(graphmet_conv_continuous_1_1_num_batches_tracked, sizeof(float), 1, f);
    fclose(f);

    f = fopen("graphnet.output.0.weight.bin", "r");
    fread(graphmet_output_0_weight, sizeof(float), 512, f);
    fclose(f);

    f = fopen("graphnet.output.0.bias.bin", "r");
    fread(graphmet_output_0_bias, sizeof(float), 16, f);
    fclose(f);

    f = fopen("graphnet.output.2.weight.bin", "r");
    fread(graphmet_output_2_weight, sizeof(float), 16, f);
    fclose(f);

    f = fopen("graphnet.output.2.bias.bin", "r");
    fread(graphmet_output_2_bias, sizeof(float), 1, f);
    fclose(f);

    // for( int i = 0; i < 16; i++ )
    // {
    //     for ( int j = 0; j < 32; j++ )
    //     {
    //         printf("%f\t", graphmet_output_0_weight[i][j]);
    //     }
    //     printf("\n");
    // }
}

// int main(void)
// {
//     float graphmet_embed_pdgid_weight[7][8];

//     FILE* f;

//     f = fopen("graphnet.embed_pdgid.weight.bin", "r");

//     fread(graphmet_embed_pdgid_weight, sizeof(float), 56, f);

//     for( int i = 0; i < 7; i++ )
//     {
//         for ( int j = 0; j < 8; j++ )
//         {
//             printf("%f\t", graphmet_embed_pdgid_weight[i][j]);
//         }
//         printf("\n");
//     }

//     printf("\n");

//     fclose(f);

//     // int array[6][3];
//     // int elt = 0;

//     // for (int i = 0; i < 5; i++){
//     //     for (int j = 0; j < 3; j++){
//     //         array[i][j] = elt;
//     //         elt++;
//     //     }
//     // }

//     // for (int i = 0; i < 5; i++){
//     //     for (int j = 0; j < 3; j++){
//     //         printf("%d\t", array[i][j]);
//     //     }
//     //     printf("\n");
//     // }

//     // printf("\n");

//     // printf("%p", array[0]);

//     // return 0;
// }