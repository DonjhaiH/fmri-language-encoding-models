import torch
import torch.nn as nn


def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    """Apply random masking for Masked Language Modeling (MLM).

    For each non-padding token, independently with probability mlm_prob:
      - 80 % of the time: replace with [MASK]
      - 10 % of the time: replace with a random vocabulary token
      - 10 % of the time: leave unchanged

    Labels are set to -100 (ignored by CrossEntropyLoss) for all positions
    that are NOT selected for prediction, so the loss is only computed on
    the masked subset.

    Args:
        input_ids     : LongTensor of shape (batch, seq_len)
        vocab_size    : int — size of the tokenizer vocabulary
        mask_token_id : int — tokenizer.mask_token_id
        pad_token_id  : int — tokenizer.pad_token_id
        mlm_prob      : float — probability of masking each token (default 0.15)

    Returns:
        masked_input_ids : LongTensor (batch, seq_len) — input with masked tokens
        labels           : LongTensor (batch, seq_len) — original ids at masked
                           positions, -100 elsewhere
    """
    labels = input_ids.clone()

    # Build per-token masking probabilities; never mask [PAD]
    prob_matrix = torch.full(input_ids.shape, mlm_prob,
                             dtype=torch.float, device=input_ids.device)
    prob_matrix[input_ids == pad_token_id] = 0.0

    # Boolean mask: which tokens are selected for prediction
    selected = torch.bernoulli(prob_matrix).bool()
    labels[~selected] = -100  # loss ignores non-selected positions

    masked_ids = input_ids.clone()

    # 80 % of selected → replace with [MASK]
    mask_indices = selected & (torch.rand_like(prob_matrix) < 0.8)
    masked_ids[mask_indices] = mask_token_id

    # Of the remaining 20 %, half (10 % overall) → random token
    remaining = selected & ~mask_indices
    random_indices = remaining & (torch.rand_like(prob_matrix) < 0.5)
    random_tokens = torch.randint(
        vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device
    )
    masked_ids[random_indices] = random_tokens[random_indices]

    # The other half of remaining (10 % overall) → unchanged (already copied)
    return masked_ids, labels


def train_bert(model, dataloader, tokenizer, epochs=3, lr=5e-5, device='cuda'):
    """Manual training loop for BERT MLM.

    For full-scale training on Bridges2, prefer lab3.2_train_bert_mlm.py which
    uses the HuggingFace Trainer (gradient accumulation, checkpointing,
    fp16, etc.).  This function is useful for quick debugging or when
    the Trainer API is not available.

    Args:
        model      : BertForMaskedLM
        dataloader : DataLoader yielding dicts with 'input_ids' and
                     'attention_mask' keys (LongTensors, batch-first)
        tokenizer  : BertTokenizerFast (for mask/pad IDs and vocab size)
        epochs     : int — number of passes over the dataset
        lr         : float — AdamW learning rate
        device     : str — 'cuda' or 'cpu'

    Returns:
        model — updated in-place and returned for convenience
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    vocab_size = tokenizer.vocab_size
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_ids, labels = mask_tokens(
                input_ids, vocab_size, mask_id, pad_id
            )

            outputs = model(
                input_ids=masked_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.4f}")

    return model
