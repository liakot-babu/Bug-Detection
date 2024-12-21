import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Preprocessing function
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that Bug1 and Bug2 columns are strings and handle missing values.
    """
    data["Bug1"] = data["Bug1"].fillna("").astype(str)
    data["Bug2"] = data["Bug2"].fillna("").astype(str)
    return data


# Dataset Class
class BugReportDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = preprocess_data(data)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        encoding = self.tokenizer(
            row["Bug1"],
            row["Bug2"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["Label"], dtype=torch.long),
        }


# DataLoader Function
def create_data_loader(data, tokenizer, batch_size, max_len=128):
    dataset = BugReportDataset(data, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model Class
class BugDuplicateDetector(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", freeze_layers=8):
        super(BugDuplicateDetector, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.domain_adapter = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.freeze_bert_layers(freeze_layers)

    def freeze_bert_layers(self, num_layers):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        domain_features = self.domain_adapter(pooled_output)
        logits = self.classifier(domain_features)
        return logits


# Trainer Class
class TransferLearningTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def train(
        self, train_loader, val_loader, epochs=3, warmup_steps=100, base_lr=2e-5, lr_decay=0.95
    ):
        optimizer_grouped_parameters = self.get_layer_specific_lr(base_lr, lr_decay)
        optimizer = AdamW(optimizer_grouped_parameters)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_val_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Metrics: {val_metrics}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")

    def get_layer_specific_lr(self, base_lr, lr_decay):
        optimizer_grouped_parameters = []
        for idx, layer in enumerate(self.model.bert.encoder.layer):
            lr = base_lr * (lr_decay ** (12 - idx))
            optimizer_grouped_parameters.append({"params": layer.parameters(), "lr": lr})
        optimizer_grouped_parameters.append(
            {"params": self.model.domain_adapter.parameters(), "lr": base_lr * 2}
        )
        optimizer_grouped_parameters.append(
            {"params": self.model.classifier.parameters(), "lr": base_lr * 4}
        )
        return optimizer_grouped_parameters

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_labels, all_preds = [], []

        for batch in data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = classification_report(all_labels, all_preds, output_dict=True)
        return avg_loss, {
            "accuracy": metrics["accuracy"],
            "precision": metrics["macro avg"]["precision"],
            "recall": metrics["macro avg"]["recall"],
            "f1": metrics["macro avg"]["f1-score"],
        }


# Main Function
def main():
    # Load and preprocess the dataset
    file_path = "cleaned_pairs.csv"
    data = pd.read_csv(file_path)
    data = preprocess_data(data)

    # Split dataset
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BugDuplicateDetector().to(device)

    train_loader = create_data_loader(train_data, tokenizer, batch_size=16)
    val_loader = create_data_loader(val_data, tokenizer, batch_size=16)
    test_loader = create_data_loader(test_data, tokenizer, batch_size=16)

    trainer = TransferLearningTrainer(model, tokenizer, device)
    trainer.train(train_loader, val_loader)

    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    main()
