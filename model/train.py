from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.nn import functional
import torch
from os import system
from torch import nn
from torch import optim
from util import init_transforms, init_train_dataset, init_val_dataset, init_test_dataset, init_resnet
from model import SimpleNet, DenseNet


def main():
    trans = init_transforms()
    batch_size = 16
    train_loader = DataLoader(init_train_dataset(trans), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(init_val_dataset(trans), batch_size=20)
    test_loader = DataLoader(init_test_dataset(trans), batch_size=20)

    model = SimpleNet()

    trainer = Trainer(model, nn.CrossEntropyLoss(), epochs=30)
    trainer.fit(train_loader, val_loader)
    trainer.test(test_loader)


class Trainer():
    def __init__(self, model, loss_fn, epochs=20) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = "cpu"
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)

    def fit(self, train_loader, val_loader):
        best_accuracy = 50
        best_loss = 2.0

        for epoch in range(self.epochs):
            training_loss = self.training_epoch(train_loader)
            validation_loss, accuracy = self.validation_epoch(val_loader)

            accuracy = int(accuracy*100)
            result = """
===========    Epoch {}   ============

       Training Loss:   {:.2f}  
       Validation Loss: {:.2f}
       Accuracy:        {} %

=====================================
""".format(epoch, training_loss, validation_loss, accuracy)

            system("clear")
            print(result)

            is_best_acc = accuracy > best_accuracy
            is_best_loss = training_loss < best_loss and accuracy > 60
            is_last_epoch = epoch != self.epochs - 1

            if is_best_acc or is_best_loss and not is_last_epoch:
                self.save_model(accuracy, validation_loss)
                best_accuracy = accuracy

    def save_model(self, accuracy, loss):
        print("Saving model...")
        loss = int(loss*100)
        torch.save(self.model, f"./saved_models/{self.model}_{accuracy}_{loss}")

    def training_epoch(self, train_loader):
        self.model.train()

        training_loss = 0.0
        for batch in train_loader:
            training_loss += self.training_step(batch)

        return training_loss / len(train_loader.dataset)

    def training_step(self, batch):
        self.optimizer.zero_grad()

        inputs, targets = batch
        loss, _ = self.evaluate(inputs, targets)
        loss.backward()

        self.optimizer.step()

        return loss.data.item()*inputs.size(0)

    def evaluate(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        output = self.model(inputs)

        correct = torch.eq(
            torch.max(functional.softmax(output, dim=-1), dim=1)[1],
            targets
        ).view(-1)

        return self.loss_fn(output, targets), correct

    def validation_epoch(self, val_loader):
        self.model.eval()

        valid_loss = 0.0
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            loss, correct = self.evaluate(inputs, targets)

            valid_loss += loss.data.item()*inputs.size(0)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

        valid_loss /= len(val_loader.dataset)
        accuracy = num_correct/num_examples

        return (valid_loss, accuracy)

    def test(self, test_loader):
        test_loss, accuracy = self.validation_epoch(test_loader)
        accuracy = int(accuracy*100)

        result = """
=========== Final Results ===========

           Test Loss: {:.2f}
           Accuracy:  {} %

=====================================
            """.format(test_loss, accuracy)

        system("clear")
        print(result)
        self.save_model(accuracy, test_loss)

if __name__ == "__main__":
    main()
