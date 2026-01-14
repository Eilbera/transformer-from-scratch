from ast import mod
#!/usr/bin/env python3
# buildChatGpt.py
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
torch.manual_seed(1337)
#max content len models sees
block_size = 256
#how many seq processed in parallel
batch_size = 32
# total training iter
num_steps = 50000
#tokens to generate in generate()
max_token = 1000

max_iter = 1000
#each step update size
lr = 1e-4
# how many batches to use when estimating loss
eval_iters = 200
#embedding dimention
n_emb = 768
# num atten heads
n_heads = 12
# depth - tranformer layer stacked
n_layers = 8
#dim per head
head_size = n_emb // n_heads

device = 'cuda' if torch.cuda.is_available() else 'cpu'
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "input.txt")
with open('input.txt', 'r', encoding = 'UTF-8' ) as f:
    text = f.read()
#1B charaters
chars = set(text)
vocab_size = len(chars)
chars = (list(chars))
final_chars = sorted(list(chars))
print("chars", final_chars)

#want dict for fast lookup of key-value pairs
stringtoindex = {ch:i for i,ch in enumerate(chars)}
indextostring = {i:ch for i,ch in enumerate(chars)}

#to encode a string of text
def encoder(text):
    results = []
    for ch in text:
        results.append(stringtoindex[ch])
    return results

def decoder(text):
    result = []
    for ch in text:
        result.append(indextostring[ch])
    return ''.join(result)

#takes the list of integers and converts to tensor to train the model
combined = torch.tensor(encoder(text), dtype=torch.long)
combined.shape

#test and validation data set
n = int(len(combined) * 0.90)
train_data = combined[:n]
test_data = combined[n:]


def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = test_data
    z = torch.randint(0, len(data)-block_size, (batch_size,))
    seq_x = []
    seq_y = []
    for i in z:
        content = data[i : i + block_size]
        seq_x.append(content)
    for i in z:
        targ = data[i+1 : i + block_size + 1]
        seq_y.append(targ)
    x = torch.stack(seq_x)
    y = torch.stack(seq_y)

    return x.to(device),y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = m(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


class bigramLanguaugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.positional_encoding_table = nn.Embedding(max_iter, n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "Wk": nn.Linear(n_emb, n_emb, bias=False),
                "Wq": nn.Linear(n_emb, n_emb, bias=False),
                "Wv": nn.Linear(n_emb, n_emb, bias=False),
                "W0": nn.Linear(n_emb, n_emb, bias=False),

                "feed_forward": nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.ReLU(),
                    nn.Linear(4 * n_emb, n_emb),
                ),
                "ln1": nn.LayerNorm(n_emb),
                "ln2": nn.LayerNorm(n_emb),
            })
            for _ in range(n_layers)
        ])

        self.apply(self._init_weights)


    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, target = None):

        token_emb = self.token_embedding_table(idx)

        T = token_emb.shape[1]
        pos = torch.arange(T, device=device)
        pos_emb = self.positional_encoding_table(pos)
        token_emb = token_emb + pos_emb

        mask = torch.tril(torch.ones(T, T, device=device))

        for layer in self.layers:
            # Get layer-specific weights and feedforward/net/ln modules
            Wk = layer["Wk"]
            Wq = layer["Wq"]
            Wv = layer["Wv"]
            W0 = layer["W0"]
            feed_forward = layer["feed_forward"]
            ln1 = layer["ln1"]
            ln2 = layer["ln2"]

            B, T, C = token_emb.shape

            # Compute Q, K, V
            K = Wk(token_emb)
            Q = Wq(token_emb)
            V = Wv(token_emb)

            # multi-head partition
            K = K.view(B, T, n_heads, head_size).transpose(1, 2)
            Q = Q.view(B, T, n_heads, head_size).transpose(1, 2)
            V = V.view(B, T, n_heads, head_size).transpose(1, 2)

            # Scaled Dot-Product Attention with masking
            attention = Q @ K.transpose(-2, -1)
            attention = attention / (K.shape[-1] ** 0.5)
            attention = attention.masked_fill(mask == 0, float('-inf'))
            attention = attention.softmax(dim=-1)
            attention = attention @ V

            # Collapsing heads
            attention = attention.transpose(1, 2).contiguous().view(B, T, C)
            attention = W0(attention)

            # Residual connection + layer norm
            token_emb = ln1(token_emb + attention)

            # Feedforward sublayer + residual, norm
            ffn_out = feed_forward(token_emb)
            token_emb = ln2(token_emb + ffn_out)


        logits = self.lm_head(token_emb)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_token):
        for _ in range(max_token):
            logits, _ = self(idx)
            logits = logits[:, -1,:]
            probs = torch.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim = 1)
        return idx


m = bigramLanguaugeModel().to(device)
xb, yb = get_batch("train")

logit,loss = m(xb,yb)
optimizer = torch.optim.AdamW(m.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_steps, eta_min = 1e-5)

best_val_loss = float('inf')
patience_counter = 0

#training
for step in range(num_steps):
    if step % eval_iters == 0:
        loss = estimate_loss()
        train_loss = loss['train']
        val_loss = loss['test']
        print(f"step {step} train loss {loss['train']} test loss {loss['test']}")
        print("sample of the model's output:")
        context = torch.zeros((1,1), dtype = torch.long, device = device)
        generated_text = m.generate(context, max_token = 500)
        print(decoder(generated_text[0].tolist()))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"new best: {best_val_loss}")
        else:
            patience_counter += 1
            print("no improvement")
            if patience_counter >= 5:
                print("early stopping")
                break
        print(f"best validation loss: {best_val_loss}")
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()
    scheduler.step()


out = m.generate(idx = torch.zeros((1,1), dtype = torch.long, device = device), max_token=max_iter)
out = out[0]
print(decoder(out.tolist()))