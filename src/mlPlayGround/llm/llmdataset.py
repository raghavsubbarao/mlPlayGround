import torch
from torch.utils.data import Dataset


class llmDataset(Dataset):

    def __init__(self, txt, tokenizer, maxLength, stride=1.):

        # Tokenize the entire text
        tokenIds = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Sliding window to chunk text into overlapping sequences of maxLength
        chunks = [(torch.tensor(tokenIds[i:i + maxLength]),  # input chunk
                   torch.tensor(tokenIds[i + 1: i + maxLength + 1]))  # target chunk
                  for i in range(0, len(tokenIds) - maxLength, stride)]

        self.__inputIds, self.__targetIds = zip(*chunks)

    @property
    def inputIds(self):
        return self.__inputIds

    @property
    def targetIds(self):
        return self.__targetIds

    def __len__(self):
        return len(self.inputIds)

    def __getitem__(self, idx):
        return self.inputIds[idx], self.targetIds[idx]


if __name__ == "__main__":
    import tiktoken
    from torch.utils.data import DataLoader

    fname = "../../../data/llmbuild/the-verdict.txt"
    with open(fname, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    ds = llmDataset(raw_text, tokenizer, 10, 5)

    # Create data loader
    dl = DataLoader(ds, batch_size=32, shuffle=True,
                    drop_last=True, num_workers=0)

