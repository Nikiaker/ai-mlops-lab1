import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
import wandb
import optuna
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# Dataset & DataLoader
class Data(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        datasets.MNIST(root="data", train=True, download=True)
        datasets.MNIST(root="data", train=False, download=True)

    def setup(self, stage=None):
        dataset = datasets.MNIST(root="data", train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(dataset, [55000, 5000])
        self.test_set = datasets.MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


# Lightning Model
class LightningModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):  # ✅ Add this method
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()  # Calculate accuracy
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)



# Hyperparameter Optimization with Optuna
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    data_module = Data(batch_size=batch_size)
    model = LightningModel(learning_rate=learning_rate)
    trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    # Model Monitoring with WandB
    wandb_logger = None
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    # Run Hyperparameter Optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)
    best_params = study.best_params

    # Train Best Model
    best_model = LightningModel(learning_rate=best_params["learning_rate"])
    best_data_module = Data(batch_size=best_params["batch_size"])
    trainer = pl.Trainer(max_epochs=2, callbacks=[checkpoint_callback])  # No WandB

    trainer.fit(best_model, best_data_module)

    # Test Model
    trainer.test(best_model, dataloaders=best_data_module.test_dataloader())
