import torch
import model
from model import ProteinCLIP, evaluate, calc_metrics
from data_utils import get_dataloaders
import config
import sys
import os


def tester():
    print(f"TEST METRICS:")
    sys.stdout.flush()

    # Load best model
    best_ckpt_path = os.path.join(config.PATH, 'best_model.pth')
    model = ProteinCLIP().load_state_dict(torch.load(best_ckpt_path))
    model = model.to(device)
    model.eval()

    # Load testing data
    _, _, test_dataloader = get_dataloaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Calculate metrics using best model
    test_preds, test_labels = evaluate(model, test_dataloader, device)
    test_metrics = calc_metrics(test_preds, test_labels)
    print(f"Accuracy: {test_metrics[0]:.4f}")
    print(f"Precision: {test_metrics[1]:.4f}")
    print(f"Recall: {test_metrics[2]:.4f}")
    print(f"F1 Score: {test_metrics[3]:.4f}")
    print(f"ROC AUC: {test_metrics[4]:.4f}")
    print(f"\n")
    sys.stdout.flush()


if __name__ == "__main__":
    tester()