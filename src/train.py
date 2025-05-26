import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

from loguru import logger

def train(config, model, data):
    """Train a GNN model with learning rate scheduling and early stopping.
    """
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get('scheduler_patience', 10))
    epochs = config['epochs']
    
    # early stopping
    patience = config.get('early_stopping_patience', 10)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs + 1):
        # train
        model.train()
        optimizer.zero_grad()
        out, _ = model((data.x, data.edge_index))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_out, _ = model((data.x, data.edge_index))
            val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])

        scheduler.step(val_loss)

        if epoch % 100 == 0:
            preds = out[data.train_mask].argmax(dim=1).cpu().numpy()
            targets = data.y[data.train_mask].cpu().numpy()
            f1 = f1_score(targets, preds, average='macro')
            val_preds = val_out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_targets = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train F1: '
                  f'{f1*100:>6.2f}% | Val Loss: {val_loss:.3f} | '
                  f'Val F1: {val_f1*100:.2f}%')

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after epoch {epoch}. Best Val Loss: {best_val_loss:.3f}')
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        logger.info("Early stopping condition never met; returning last model state.")

    return model

@torch.no_grad()
def test(model, data, mask=None):
    """Evaluate the model on a given mask and return precision, recall, and f1.
    """
    model.eval()
    out, _ = model((data.x, data.edge_index))
    if mask is None:
        mask = data.test_mask
    preds = out[mask].argmax(dim=1).cpu().numpy()
    targets = data.y[mask].cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    return precision, recall, f1