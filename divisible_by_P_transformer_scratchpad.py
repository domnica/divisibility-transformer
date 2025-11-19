import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm

# Configuration
BIT_LENGTH = 32
P = 3 # The divisor (Number of classes/remainders)
N_SAMPLES = 100000
BATCH_SIZE = 128 # Increased batch size slightly for efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. New Data Generation (Inputs and Scratchpad Targets)
def generate_scratchpad_data(num_samples, bit_length=32, P=3):
    """Generates data including the intermediate remainders (Scratchpad)."""
    max_int = (1 << bit_length) - 1
    rng = np.random.default_rng(42)
    # Start numbers from 0
    numbers = rng.integers(0, high=max_int + 1, size=num_samples, dtype=np.int64)

    inputs = []
    targets = []
    # Final labels are used only for balancing the dataset
    final_labels = (numbers % P).astype(int)

    format_spec = f'0{bit_length}b'
    for n in numbers:
        bin_str = format(n, format_spec)
        # MSB First Input Sequence
        input_seq = [int(digit) for digit in bin_str]
        inputs.append(input_seq)

        # Calculate intermediate remainders (The Scratchpad Target)
        target_seq = []
        current_remainder = 0
        for bit in input_seq:
            # The iterative algorithm: (R*2 + bit) % P
            current_remainder = (current_remainder * 2 + bit) % P
            target_seq.append(current_remainder)
        targets.append(target_seq)

    return inputs, targets, final_labels

# 2. Model Architecture (Causal Transformer predicting sequence)
class ScratchpadTransformer(nn.Module):
    def __init__(self, bit_length, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.0):
        super(ScratchpadTransformer, self).__init__()
        self.d_model = d_model
        self.bit_length = bit_length

        # Input Embedding (Bits 0, 1)
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        # Learned Positional Embeddings
        self.pos_embedding = nn.Embedding(num_embeddings=bit_length, embedding_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Causal Transformer (using Encoder with Mask)
        # norm_first=True (Pre-LN) for stability
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output Head (Predicts remainders 0, 1, 2)
        # CHANGE: This head is now applied to every position, not just the last one.
        self.output_head = nn.Linear(d_model, num_classes)

        # Causal Mask Registration
        self.register_buffer("causal_mask", self.generate_causal_mask(bit_length))

    @staticmethod
    def generate_causal_mask(size):
        """Generates an additive causal mask (blocks future positions)."""
        # 0 allows attention, -inf blocks it.
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src):
        # Input src shape: (Batch, SeqLen)

        # Apply embeddings and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        positions = torch.arange(0, self.bit_length, dtype=torch.long, device=src.device)
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(x + pos_emb)

        # Pass through the Transformer encoder WITH THE CAUSAL MASK
        output = self.transformer_encoder(x, mask=self.causal_mask)
        # output shape: (Batch, SeqLen, D_Model)

        # CHANGE: Predict the remainder at EVERY position
        logits = self.output_head(output)
        # logits shape: (Batch, SeqLen, NumClasses)
        return logits

# --- Data Preparation ---
X_data, Y_targets, Y_labels = generate_scratchpad_data(N_SAMPLES, BIT_LENGTH, P)

# Balance the dataset based on the FINAL remainder (Y_labels)
indices_per_class = []
for i in range(P):
    indices_per_class.append(np.where(Y_labels == i)[0])

min_class_size = min(len(indices) for indices in indices_per_class)

rng = np.random.default_rng(123)
balanced_indices = []
for indices in indices_per_class:
    sampled_indices = rng.choice(indices, size=min_class_size, replace=False)
    balanced_indices.append(sampled_indices)

balanced_idx = np.concatenate(balanced_indices)
rng.shuffle(balanced_idx)

X_data = [X_data[i] for i in balanced_idx]
Y_targets = [Y_targets[i] for i in balanced_idx]

print(f"\nGenerated {len(X_data)} samples (after balancing).")

# Convert data to PyTorch Tensors
X_tensor = torch.tensor(X_data, dtype=torch.long)
# Y_targets (the scratchpad) are the labels for training, type Long for CrossEntropy
Y_tensor = torch.tensor(Y_targets, dtype=torch.long)

# Split and Create DataLoaders
# We stratify based on the final label, extracted from the last element of Y_tensor
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, Y_tensor, test_size=0.2, random_state=42, stratify=Y_tensor[:,-1]
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---- Neural network part -----

# Hyperparameters (We keep the architecture robust)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.0         # Zero dropout is crucial for precise algorithmic learning
LEARNING_RATE = 0.002
EPOCHS = 60           # Often converges much faster with scratchpad

# Initialize the model
model = ScratchpadTransformer(
    bit_length=BIT_LENGTH,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    num_classes=P,
    dropout=DROPOUT
).to(DEVICE)

# Loss and Optimizer
# We use CrossEntropyLoss, which will average the loss across all sequence positions
criterion = nn.CrossEntropyLoss()
# CHANGE: Explicitly set weight_decay=0.0. This regularization often hinders algorithmic learning.
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

# Scheduler Setup (Warmup + Linear Decay)
WARMUP_STEPS_RATIO = 0.1
TOTAL_TRAINING_STEPS = len(train_loader) * EPOCHS
WARMUP_STEPS = int(TOTAL_TRAINING_STEPS * WARMUP_STEPS_RATIO)

def lr_lambda(current_step: int):
    if current_step < WARMUP_STEPS:
        # Linear Warmup
        return float(current_step) / float(max(1, WARMUP_STEPS))
    # Linear Decay
    num_decay_steps = TOTAL_TRAINING_STEPS - WARMUP_STEPS
    progress = (current_step - WARMUP_STEPS) / max(1, num_decay_steps)
    return max(0.0, 1.0 - progress)

scheduler = LambdaLR(optimizer, lr_lambda)

print("\nModel Initialized.")
# The warning about nested tensors when norm_first=True can be ignored if it appears.
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- Training Loop ---
print("\nStarting Training...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    # We will track the accuracy of the FINAL prediction specifically
    correct_train_final = 0
    total_train = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for batch_X, batch_y in progress_bar:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        # batch_X shape: (Batch, SeqLen) - Inputs
        # batch_y shape: (Batch, SeqLen) - Target Scratchpad

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        # outputs shape: (Batch, SeqLen, NumClasses)

        # CHANGE: Calculate loss across the entire sequence
        # We must reshape the outputs and targets for CrossEntropyLoss
        # Reshape outputs to (Batch * SeqLen, NumClasses)
        # Reshape batch_y to (Batch * SeqLen)
        loss = criterion(outputs.view(-1, P), batch_y.view(-1))
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
        optimizer.step()
        scheduler.step()

        # Calculate accuracy (Focus on the final remainder)
        preds = outputs.argmax(dim=2) # (Batch, SeqLen)
        # Get the prediction and target for the last position (-1)
        final_preds = preds[:, -1]
        final_targets = batch_y[:, -1]
        correct_train_final += (final_preds == final_targets).sum().item()
        total_train += batch_y.size(0)

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix(loss=loss.item(), final_acc=correct_train_final/total_train, lr=f"{current_lr:.6f}")

    avg_train_loss = epoch_loss / len(train_loader)
    train_accuracy_final = correct_train_final / total_train

    # --- Evaluation Loop ---
    model.eval()
    test_correct_final = 0
    test_total = 0
    # Also track sequence accuracy (if the entire scratchpad is correct)
    test_correct_seq = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=2)

            # Evaluate accuracy of the final position
            final_preds = preds[:, -1]
            final_targets = batch_y[:, -1]
            test_correct_final += (final_preds == final_targets).sum().item()
            test_total += batch_y.size(0)

            # Evaluate sequence accuracy
            # Check if all predictions in the sequence match the targets
            seq_acc = (preds == batch_y).all(dim=1).sum().item()
            test_correct_seq += seq_acc

    test_accuracy_final = test_correct_final / test_total
    test_accuracy_seq = test_correct_seq / test_total

    print(f"Epoch {epoch+1} Finished. Train Loss: {avg_train_loss:.4f}, Train Final Acc: {train_accuracy_final:.4f}, Test Final Acc: {test_accuracy_final:.4f}, Test Seq Acc: {test_accuracy_seq:.4f}")

print("Training Finished.")
