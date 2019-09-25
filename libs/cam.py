import torch
import torch.nn.functional as F


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class GradCAM(object):
    """ Grad CAM for video captioning """

    def __init__(self, encoder, decoder, target_layer):
        super().__init__()
        """
        Args:
            encoder: feature encoder
            decoder: output captions
            target_layer: conv_layer you want to visualize
        """

        self.encoder = encoder
        self.decoder = decoder
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def __call__(self, feature, caption, length):
        """
        Args:
            feature: input feature. shape =>(1, C, T, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # caption generation
        out = self.encoder(feature)
        out = self.decoder(out, caption, length)

        # sum the score of predicted words
        scores, _ = torch.max(out, dim=1)
        score = scores.sum()

        # caluculate cam of the predicted caption
        cam = self.getGradCAM(self.values, score)

        return cam

    def getGradCAM(self, values, score):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, T, H, W)
        score: the output of the model before softmax
        cam: class activation map.  shape=> (1, 1, T, H, W)
        '''

        self.encoder.zero_grad()
        self.decoder.zero_grad()

        score.backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients

        n, c, _, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class GradCAMpp():
    """ Grad CAM plus plus for video captioning """

    def __init__(self, encoder, decoder, target_layer):
        super().__init__()
        """
        Args:
            encoder: feature encoder
            decoder: output captions
            target_layer: conv_layer you want to visualize
        """

        self.encoder = encoder
        self.decoder = decoder
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def __call__(self, feature, caption, length):
        """
        Args:
            feature: input feature. shape =>(1, C, T, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # caption generation
        out = self.encoder(feature)
        out = self.decoder(out, caption, length)

        # sum the score of predicted words
        scores, _ = torch.max(out, dim=1)
        score = scores.sum()

        cam = self.getGradCAMpp(self.values, score)

        return cam

    def getGradCAMpp(self, values, score):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, T, H, W)
        score: the output of the model before softmax. shape
        cam: class activation map.  shape=> (1, 1, T, H, W)
        '''

        self.encoder.zero_grad()
        self.decoder.zero_grad()

        score.backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients

        n, c, _, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(
            -1, keepdim=True).view(n, c, 1, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score.exp() * gradients)
        weights = alpha * relu_grad
        weights = weights.view(n, c, -1).sum(-1).view(n, c, 1, 1, 1)

        # shape => (1, 1, T', H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data
