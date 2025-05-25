24-05-2025
19:58

Status:
Tags:[[2 - Tags/Transformer|Transformer]], [[Embeddings]]
# Transformer

## Attention Mechanism
Attention mechanism in time series
![[Pasted image 20250524203949.png]]

The basic plan for attention in text. Proximity based context (like in time series) is not suitable for text, it needs grammatical context.
![[Pasted image 20250524204413.png]]

The embeddings: $V_{K}$ Vector Embedding of King, $V_{Q}$ Vector Emb of Queen,
i.e. the embeddings capture the structure which can then be used to provide the re-weighing scheme
$$
W_{KQ}=V_{K}.V_{Q} \quad or \quad V_{K}^TV_{Q}
$$

![[Pasted image 20250524205016.png]]

This is called Self Attention: 
- We haven't trained any weights, 
- Proximity is not the primary thing
- Shape independent

![[Pasted image 20250524205454.png]]



## Understanding the Transformer


# References
[https://goyalpramod.github.io/blogs/Transformers_laid_out]
