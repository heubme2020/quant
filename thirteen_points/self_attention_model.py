import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random


class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # feature num

        self.W_Q = nn.Linear(self.D, self.D * self.H)
        self.W_K = nn.Linear(self.D, self.D * self.H)
        self.W_V = nn.Linear(self.D, self.D * self.H)
        self.W_O = nn.Linear(self.D * self.H, self.D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x):
        Q = self.W_Q(x)  # (B, S, D)
        K = self.W_K(x)  # (B, S, D)
        V = self.W_V(x)  # (B, S, D)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_weights, V)  # (B, H, S, D)
        output = self.concat_heads(output)  # (B, S, D*H)
        output = self.W_O(output)
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        attention_output = self.attention(x)
        output = x + attention_output
        output = self.linear(output)
        output = self.norm(output)
        output = self.gelu(output)
        output = self.out(output)
        return output


class Decoder(nn.Module):
    ##input_shape, output_shape这里[feature_num, seq_length]
    def __init__(self, input_shape, output_shape, device=None):
        super(Decoder, self).__init__()
        self.device = device
        self.in_feature_num = input_shape[0]
        self.in_seq_length = input_shape[1]
        self.out_feature_num = output_shape[0]
        self.out_seq_length = output_shape[1]

        self.linear = nn.Linear(input_shape[0]*input_shape[1], input_shape[0]*output_shape[1])
        self.norm = nn.LayerNorm(input_shape[0]*output_shape[1])
        self.gelu = nn.GELU()
        self.linear_out = nn.Linear(input_shape[0]*output_shape[1], output_shape[0]*output_shape[1])


    def forward(self, x):
        batch_size = x.shape[0]
        decoder_output = x.reshape(batch_size, self.in_feature_num*self.in_seq_length)
        decoder_output = self.linear(decoder_output)
        decoder_output = self.norm(decoder_output)
        decoder_output = self.gelu(decoder_output)
        output = self.linear_out(decoder_output)
        output = output.view(batch_size, self.out_seq_length, self.out_feature_num)
        return output


class Growth_Death(nn.Module):
    def __init__(self, input_shape,  output_shape):
        super(Growth_Death, self).__init__()
        self.first_encoder = Encoder(input_shape[0], 8)
        self.first_decoder = Decoder(input_shape, output_shape)
        self.second_encoder = Encoder(output_shape[0], 4)
        self.second_decoder = Decoder([output_shape[0], 23], [2, 1])

    def forward(self, x, financial_past):
        financial_fore = self.first_encoder(x)
        financial_fore = self.first_decoder(financial_fore)
        financial = torch.cat((financial_past, financial_fore[:, :6, :]), dim=1)
        regression_label = self.second_encoder(financial)
        regression_label = self.second_decoder(regression_label)
        return financial_fore, regression_label



if __name__ == '__main__':
    model = Growth_Death([123, 1201], [101, 7]).to('cuda')
    tensor = torch.randn(64, 1201, 123).to('cuda')
    financial_past = torch.randn(64, 17, 101).to('cuda')
    financial_fore, regression_label = model(tensor, financial_past)
    print(financial_fore)
    print(regression_label)










