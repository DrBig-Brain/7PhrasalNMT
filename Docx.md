
### Docx on Training File:


configs
- We are using CPU as of now for training ,dont have gpu
- epcohs=10
- batch_size=64

- loss: Cross Entropy loss
- optimizer: Adam



--- 
Train Loop
- extract data ,x ,y ,p[positional embedding]
y (target)   = [<s>, I, am, happy, </s>]
tgt_in       = [<s>, I, am, happy]  ---> fed to model
tgt_out      = [I, am, happy, </s>]  ---> for loss
out          = model(x, tgt_in, ...) ---> predicted logits

- calculate loss
- optimizer step

return loss



## Transformer 🎃

phrase embedding 

- each phrase extracted from sentence during preprocessing gets a vector representation

- model with higher level context

- Helps the transformer better understand **meaningful chunks**









## Transformer File:


Setup the config

- vocab size
- sequence length
- n block
- n head
- embed_dim
- phrase emb_dim

use the sentence that is concatenated with phrases embeddings


- Main 🎃
– Token Embedding
– Phrase embedding : like embedding of 7 sentences
– pos embedding



- encoder layer
- decoder layer


On forward pass:

- getting shapes of target and source(y,x)
- embedding for each and concatenation
- final emb - with positional emb
- encoder layer
- preparing target embeddin gsame as above
- decoder layer(target embedding,encoder output)
- return fully connected's output

### Generate functions

source phrase : phrase indices for source
set eval() -> disables gradient computation and dropout or norm updates

– prepare source embeddings 
: all the embeddings
– intitializw target sequence
use decoder
start generating 1 by 1 token
append

return generated sequence

