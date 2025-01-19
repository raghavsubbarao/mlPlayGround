import torch
import tiktoken
from mlPlayGround.recurrent.attention import multiHeadAttentionTorchSDP
from mlPlayGround.model import baseTorchModel, baseLossTracker

GPT2_SMALL_124M_CONFIG = {"vocabSize": 50257,     # Vocabulary size
                          "contextLength": 256,  # 1024,  # Context length
                          "embedDim": 768,        # Embedding dimension
                          "nHeads": 12,           # Number of attention heads
                          "nLayers": 12,          # Number of layers
                          "dropout": 0.1,         # Dropout rate
                          "bias": False           # Query-Key-Value bias
                          }

GPT2_MEDIUM_355M_CONFIG = {"vocabSize": 50257,     # Vocabulary size
                           "contextLength": 1024,  # Context length
                           "embedDim": 1024,       # Embedding dimension
                           "nHeads": 16,           # Number of attention heads
                           "nLayers": 24,          # Number of layers
                           "dropout": 0.1,         # Dropout rate
                           "bias": False           # Query-Key-Value bias
                           }

GPT2_LARGE_774M_CONFIG = {"vocabSize": 50257,     # Vocabulary size
                          "contextLength": 1024,  # Context length
                          "embedDim": 1280,       # Embedding dimension
                          "nHeads": 20,           # Number of attention heads
                          "nLayers": 36,          # Number of layers
                          "dropout": 0.1,         # Dropout rate
                          "bias": False           # Query-Key-Value bias
                          }

GPT2_XLARGE_1558M_CONFIG = {"vocabSize": 50257,     # Vocabulary size
                            "contextLength": 1024,  # Context length
                            "embedDim": 1600,       # Embedding dimension
                            "nHeads": 25,           # Number of attention heads
                            "nLayers":48,          # Number of layers
                            "dropout": 0.1,         # Dropout rate
                            "bias": False           # Query-Key-Value bias
                            }

class gptTransformerBlock(torch.nn.Module):

    def __init__(self, embedDim, contextLength, nHeads, dropout, bias=False):
        super(gptTransformerBlock, self).__init__()
        self.mha = multiHeadAttentionTorchSDP(nin=embedDim, nout=embedDim,
                                              contextLength=contextLength,
                                              nHeads=nHeads, dropout=dropout,
                                              bias=bias)

        # feed forward
        self.ff = torch.nn.Sequential(torch.nn.Linear(embedDim, 4 * embedDim),
                                      torch.nn.GELU(),
                                      torch.nn.Linear(4 * embedDim, embedDim))

        self.norm1 = torch.nn.LayerNorm(embedDim)
        self.norm2 = torch.nn.LayerNorm(embedDim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, y):
        shortcut = y  # skip connection
        y = self.norm1(y)    # layer norm: b x nTokens x embedDim
        y = self.mha(y)      # attention: b x nTokens x embedDim
        y = self.dropout(y)  # dropout: b x nTokens x embedDim
        y = y + shortcut     # Add the skip connection

        shortcut = y  # skip connection
        y = self.norm2(y)    # layer norm: b x nTokens x embedDim
        y = self.ff(y)       # feed-forward: b x nTokens x embedDim
        y = self.dropout(y)  # dropout: b x nTokens x embedDim
        y = y + shortcut     # Add the skip connection

        return y


class generativePretrainedTransformer(torch.nn.Module):
    def __init__(self, vocabSize, embedDim, contextLength, nHeads, nLayers, dropout, bias=False):
        super(generativePretrainedTransformer, self).__init__()
        self.wordEmbed = torch.nn.Embedding(vocabSize, embedDim)
        self.posnEmbed = torch.nn.Embedding(contextLength, embedDim)
        self.dropout = torch.nn.Dropout(dropout)

        # transformer blocks
        self.blocks = torch.nn.Sequential(*[gptTransformerBlock(embedDim, contextLength,
                                                                nHeads, dropout, bias)
                                            for _ in range(nLayers)])

        self.lnorm = torch.nn.LayerNorm(embedDim)
        self.ffOut = torch.nn.Linear(embedDim, vocabSize, bias=False)

    def forward(self, y):
        _, seqLen = y.shape
        wordEmbeds = self.wordEmbed(y)
        posnEmbeds = self.posnEmbed(torch.arange(seqLen, device=y.device))
        x = wordEmbeds + posnEmbeds
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.lnorm(x)
        return self.ffOut(x)  # return the logits


class gptLargeLanguageModel(generativePretrainedTransformer, baseTorchModel):
    def __init__(self, vocabSize, embedDim, contextLength, nHeads,
                 nLayers, dropout, bias=False, tokenizer='gpt2', **kwargs):
        super(gptLargeLanguageModel, self).__init__(vocabSize, embedDim, contextLength,
                                                    nHeads, nLayers, dropout, bias=bias,
                                                    **kwargs)

        self.__contextLength = contextLength
        self.__tokenizer = tiktoken.get_encoding(tokenizer)

        self.totalLossTracker = baseLossTracker(name="totalLoss")
        self.perplexLossTracker = baseLossTracker(name="perplexLoss")

    @property
    def contextLength(self):
        return self.__contextLength

    @property
    def tokenizer(self):
        return self.__tokenizer

    def textToToken(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        return torch.tensor(encoded).unsqueeze(0)  # add batch dimension

    def tokenToText(self, tokens):
        flat = tokens.squeeze(0)  # remove batch dimension
        return self.tokenizer.decode(flat.tolist())

    def generateText(self, tokens, nNewTokens, temperature=0.0, top_k=None, eos_id=None):
        self.eval()
        tokens = tokens.to(self.device)

        # tokens is (batch, nTokens) array of indices in the current context
        for _ in range(nNewTokens):
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            conditioningTokens = tokens[:, -self.contextLength:]

            # Get the predictions
            with torch.no_grad():
                logits = self(conditioningTokens)

            # Focus only on the last time step
            # (batch, nTokens, vocabSize) becomes (batch, vocabSize)
            logits = logits[:, -1, :]

            # Get the idx of the vocab entry with the highest logits value
            nextToken = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            tokens = torch.cat((tokens, nextToken), dim=1)  # (batch, n_tokens + 1)

        return tokens

    def trainStep(self, data, optimizer):
        X, target = data
        X = X.to(self.device)
        targets = target.to(self.device)

        logits = self.forward(X)

        totalLoss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        perplexLoss = torch.exp(totalLoss)

        # Backpropagation
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

        self.totalLossTracker.updateState(totalLoss)
        self.perplexLossTracker.updateState(perplexLoss)

        return {"totalLoss": self.totalLossTracker.result(),
                "perplexLoss": self.perplexLossTracker.result()}

    def validStep(self, data):
        X, target = data
        X = X.to(self.device)
        targets = target.to(self.device)

        logits = self.forward(X)

        totalLoss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return totalLoss


if __name__ == "__main__":

    # # Step 1
    # # test the transformer block
    # torch.manual_seed(123)  # seed
    # x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    # block = gptTransformerBlock(768, 1024, 12, 0.1)
    # output = block(x)
    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)

    # # Step 2
    # # test the gpt module
    # tokenizer = tiktoken.get_encoding("gpt2")
    #
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)
    # print(batch)
    #
    # torch.manual_seed(123)
    # model = generativePretrainedTransformer(**GPT2_SMALL_124M_CONFIG)
    #
    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # # print(logits)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params:,}")
    # print(f"Adjusted number of parameters: {total_params - 50257 * 768:,}")

    # # Step 3
    # # Test the GPT model for text generation
    # t = llmGpt(**GPT2_SMALL_124M_CONFIG)
    # start = "Every effort moves you"
    # tokens = t.textToToken(start)
    # newTokens = t.generateText(tokens, 10)
    # print(t.tokenToText(newTokens))

    # Step 4
    # Test the GPT model for simple training
    from mlPlayGround.llm.llmdataset import llmDataset
    from torch.utils.data import DataLoader

    llmConfig = GPT2_SMALL_124M_CONFIG
    model = gptLargeLanguageModel(**llmConfig)

    start = "I felt able to face"
    tokens = model.textToToken(start)
    newTokens = model.generateText(tokens, 10)
    print(model.tokenToText(newTokens))

    fname = "../../../data/llmbuild/the-verdict.txt"
    with open(fname, "r", encoding="utf-8") as f:
        text_data = f.read()

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    valid_data = text_data[split_idx:]

    trainDataset = llmDataset(train_data, model.tokenizer,
                              maxLength=llmConfig['contextLength'],
                              stride=llmConfig['contextLength'])
    validDataset = llmDataset(valid_data, model.tokenizer,
                              maxLength=llmConfig['contextLength'],
                              stride=llmConfig['contextLength'])

    # Create data loader
    trainDataloader = DataLoader(trainDataset, batch_size=2, shuffle=False,
                                 drop_last=False, num_workers=0)
    validDataloader = DataLoader(validDataset, batch_size=2, shuffle=False,
                                 drop_last=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    model.trainLoop(trainDataloader, optimizer,
                    epochs=10, reportIters=1,
                    validDataLoader=validDataloader)

    tokens = model.textToToken(start)
    newTokens = model.generateText(tokens, 10)
    print(model.tokenToText(newTokens))

    # save the model
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
               "../../../models/llm/gpt_small_test.pth"
    )

    # load the model
    checkpoint = torch.load("../../../models/llm/gpt_small_test.pth", weights_only=True)
    model = gptLargeLanguageModel(**llmConfig)
    model.load_state_dict(checkpoint["model_state_dict"])
