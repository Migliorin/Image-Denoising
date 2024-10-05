import lightning as L
import torch


class TrainModule(L.LightningModule):
    def __init__(self, model, loss, optim):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim = optim

    def forward(self, denoised_img):
        return self.model(denoised_img)

    def training_step(self, batch, batch_idx):
        ori_img, noi_img = batch

        noi_img = noi_img.cuda()
        denoised_img = self.model(noi_img)
        denoised_img = denoised_img

        loss = self.loss(denoised_img, ori_img)

        self.log("loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ori_img, noi_img = batch

        noi_img = noi_img.cuda()
        denoised_img = self.model(noi_img)
        denoised_img = denoised_img

        loss = self.loss(denoised_img, ori_img)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return self.optim
