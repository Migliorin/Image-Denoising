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

        loss = self.compute_loss_patch(denoised_img, ori_img.cpu())

        self.log("loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        ori_img, noi_img = batch

        noi_img = noi_img.cuda()
        denoised_img = self.model(noi_img)
        denoised_img = denoised_img.cpu()

        loss = self.compute_loss_patch(denoised_img, ori_img.cpu())

        self.log("val_loss", loss, prog_bar=True)

        return loss


    def compute_loss(self,denoised_img,ori_img):
        ori_img = self.model.patch_tokenization(ori_img)
        denoised_img = denoised_img[:,:-1,:]

        return self.loss(denoised_img,ori_img)

    def compute_loss_patch(self,denoised_img,ori_img):
        ori_img = self.model.patch_tokenization(ori_img)
        denoised_img = denoised_img[:,:-1,:]

        denoised_img = self.model.unpatch_tokenization(denoised_img)
        denoised_img = self.model.patch_tokenization(denoised_img)

        self.loss(denoised_img,ori_img)


    def configure_optimizers(self):
        return self.optim
