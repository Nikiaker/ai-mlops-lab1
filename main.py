import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
import wandb
import optuna
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from torch import Tensor
from typing import Any

# Dataset & DataModule
class Data(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self) -> None:
        datasets.MNIST(root="data", train=True, download=True)
        datasets.MNIST(root="data", train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        dataset = datasets.MNIST(root="data", train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(dataset, [55000, 5000])
        self.test_set = datasets.MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size)


# Lightning Model
class LightningModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Hyperparameter Optimization with Optuna
def objective(trial: optuna.Trial) -> float:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    data_module = Data(batch_size=batch_size)
    model = LightningModel(learning_rate=learning_rate)
    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_loss"].item()

def predict_image(model_checkpoint: str, image_path: str) -> int:
    model = LightningModel.load_from_checkpoint(model_checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        logits = model(tensor_image)
        prediction = logits.argmax(dim=1).item()

    print(f"Predicted digit: {prediction}")
    return prediction

if __name__ == "__main__":
    # Hyperparameter Optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params

    # Model Monitoring with WandbLogger
    wandb_logger = WandbLogger(project="ai-mlops-lab1")

    # Model Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath="checkpoints",
        filename="best_model-{epoch:02d}-{val_loss:.2f}" 
    )

    # Train best model
    best_model = LightningModel(learning_rate=best_params["learning_rate"])
    best_data_module = Data(batch_size=best_params["batch_size"])
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    trainer.fit(best_model, best_data_module)
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_checkpoint_path}")

    # Test the best model
    trainer.test(best_model, dataloaders=best_data_module.test_dataloader())
