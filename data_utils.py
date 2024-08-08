import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from esm_utils import load_esm2_model, get_latents

# Dataset class
class InteractionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, model):
        super(InteractionDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        seq1 = self.data.iloc[idx]['first_sequence']
        seq2 = self.data.iloc[idx]['second_sequence']
        label = self.data.iloc[idx]['label']

        # Get latent representations of sequences
        seq1_latents = torch.tensor(get_latents(self.model, self.tokenizer, seq1), dtype=torch.float)
        seq2_latents = torch.tensor(get_latents(self.model, self.tokenizer, seq2), dtype=torch.float)

        # Initialize attention masks of sequences
        seq1_attn_mask = torch.ones(seq1_latents.size(0), dtype=torch.float)
        seq2_attn_mask = torch.ones(seq2_latents.size(0), dtype=torch.float)

        labels = torch.tensor(label, dtype=torch.float)

        return seq1_latents, seq2_latents, seq1_attn_mask, seq2_attn_mask, labels

def collate_fn(batch):
    seq1_latents, seq2_latents, seq1_attn_mask, seq2_attn_mask, labels = zip(*batch)

    seq1_latents_padded = pad_sequence([torch.tensor(l) for l in seq1_latents], batch_first=True, padding_value=0)
    seq2_latents_padded = pad_sequence([torch.tensor(l) for l in seq2_latents], batch_first=True, padding_value=0)

    seq1_attn_mask_padded = pad_sequence([torch.tensor(m) for m in seq1_attn_mask], batch_first=True, padding_value=0)
    seq2_attn_mask_padded = pad_sequence([torch.tensor(m) for m in seq2_attn_mask], batch_first=True, padding_value=0)

    # Convert binary attention mask to boolean values
    seq1_attn_mask_padded = (seq1_attn_mask_padded != 0)
    seq2_attn_mask_padded = (seq2_attn_mask_padded != 0)

    labels = torch.tensor(labels, dtype=torch.float)
    
    return seq1_latents_padded, seq2_latents_padded, seq1_attn_mask_padded, seq2_attn_mask_padded, labels

def get_dataloaders(config):
    tokenizer, model = load_esm2_model(config.MODEL_NAME)

    train_dataset = InteractionDataset(config.TRAIN_CSV, tokenizer, model)
    val_dataset = InteractionDataset(config.VAL_CSV, tokenizer, model)
    test_dataset = InteractionDataset(config.TEST_CSV, tokenizer, model)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader