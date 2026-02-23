"""
self attention:
     Enhances the information content of an input by considering the context

This is a recreation of the scaled dot product attention mechanism
"""

import torch

sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
# print(sentence_int)

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