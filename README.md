# Arm-Keypoints-Detection
link to models saved states:
https://drive.google.com/drive/folders/14hIE8WW47MnAosOzoWhQTr09Qee44_xH?usp=sharing


class FullResnet(nn.Module):
    def __init__(self, modelA, modelB):
        super(FullResnet, self).__init__()

        self.modelA = modelA
        self.modelB = modelB


    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        #print(x.shape)
        out=x
        #print(out)
        return out
net4 = torchvision.models.alexnet(pretrained = True)
newmodel3 = torch.nn.Sequential(*(list(net4.children())[:-1]))#[1, 320, 3, 3]
