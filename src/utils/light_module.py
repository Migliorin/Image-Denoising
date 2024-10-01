import lightning as L
import torch



class LightningVisionTransformer(L.LightningModule):
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
        denoised_img = denoised_img.cpu()

        loss_ = self.loss(denoised_img, ori_img.cpu())

        self.log("loss",loss_,prog_bar=True)

        return loss_

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        ori_img, noi_img = batch

        noi_img = noi_img.cuda()
        denoised_img = self.model(noi_img)
        denoised_img = denoised_img.cpu()

        loss_ = self.loss(denoised_img, ori_img.cpu())

        self.log("val_loss", loss_,prog_bar=True)

        return loss_


    def configure_optimizers(self):
        return self.optim
