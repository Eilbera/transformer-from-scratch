# Shakespeare Transformer from Scratch

Character-level transformer language model built from scratch in PyTorch. Trained on Shakespeare's works.

Built following Andrej Karpathy’s “Let’s build GPT: from scratch, in code, spelled out” video, then scaled up and experimented with on my own.


## Architecture
- 8 transformer layers
- 12 attention heads
- 768 embedding dimensions
- 57.5M parameters 

## Results
Final validation loss: 1.54 after after 2k steps

## Learnings
- Multi-head attention took model from val loss of above 3 to around 1.54
- Proper weight initialization for deep networks
- Training dynamics and hyperparameter tuning
- Early stopping with patience

## Sample output

step 2000 train loss 1.1535645723342896 test loss 1.5418249368667603
sample of the model's output:
RURENCESO:
Poor Given, so the people are doth loss;
No monaments reasons, within the warrants warlike,
Disconditiously years in with words.

Clown:
Why the sings of boisters coar are out
Forced and amach litter fear a king.

AUTOLYCUS:
Master welcome you, simy che an shomingle ome ay
He Ist doumeldd. omay t mimeee.
imeerend on; oumee.
My. breereest Jureeechiold ome on.
I it gust, ldd omeeeee; nt on: omeeeee omee treeeeeeeest'st it n; omigust, treest, a?-------peeapeeeist n. nddupeeeeeeeeeeeeweeee
no improvement
best validation loss: 1.540079951286316
