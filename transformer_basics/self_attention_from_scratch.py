"""
self attention:
     Enhances the information content of an input by considering the context

This is a recreation of the scaled dot product attention mechanism - first using single head then multi head attention

Full credit to: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
"""

"""Single Head Implementation"""

import torch
import torch.nn.functional as F

sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])


"""
Now, using the integer-vector representation of the input sentence,
we can use an embedding layer to encode the inputs into a real-vector embedding. 
Here, we will use a 16-dimensional embedding such that each input word is represented by a 16-dimensional vector. 
Since the sentence consists of 6 words, this will result in a 6×16
"""

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

"""
Defining the Weights Matrices
"""

torch.manual_seed(123)

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 24, 24, 28

W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))

"""
Computing Unnormalized Scaled Dot Product Attention Weights
"""

x_2 = embedded_sentence[1]
query_2 = W_query.matmul(x_2)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)

keys = W_key.matmul(embedded_sentence.T).T
values = W_value.matmul(embedded_sentence.T).T

omega_2 = query_2.matmul(keys.T)

"""Computing attention scores"""

attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)

"""Determination of the context vector"""

context_vector_2 = attention_weights_2.matmul(values)

"""
Multi Head Attention of dim 3 (3 attention heads)
"""

h = 3
multihead_W_query = torch.nn.Parameter(torch.rand(h, d_q, d))
multihead_W_key = torch.nn.Parameter(torch.rand(h, d_k, d))
multihead_W_value = torch.nn.Parameter(torch.rand(h, d_v, d))

multihead_query_2 = multihead_W_query.matmul(x_2)
multihead_key_2 = multihead_W_key.matmul(x_2)
multihead_value_2 = multihead_W_value.matmul(x_2)

stacked_inputs = embedded_sentence.T.repeat(3, 1, 1)

multihead_keys = torch.bmm(multihead_W_key, stacked_inputs)
multihead_values = torch.bmm(multihead_W_value, stacked_inputs)
print("multihead_keys.shape:", multihead_keys.shape)
print("multihead_values.shape:", multihead_values.shape)

multihead_keys = multihead_keys.permute(0, 2, 1)
multihead_values = multihead_values.permute(0, 2, 1)
print("multihead_keys.shape:", multihead_keys.shape)
print("multihead_values.shape:", multihead_values.shape)