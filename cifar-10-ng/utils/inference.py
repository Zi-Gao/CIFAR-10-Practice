import torch
import torch.nn as nn
import torch.nn.functional as F

def predict(model, images, device):
    """Standard inference."""
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
    return outputs

def predict_tta(model, images, device):
    """Test-Time Augmentation: Mirror + Translate (from airbench96).
    Returns BOTH standard and TTA outputs to save compute in training loops.
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        
        # Basic mirror TTA
        outputs = model(images)
        outputs_flipped = model(torch.flip(images, [-1]))
        logits_mirror = 0.5 * (outputs + outputs_flipped)
        
        # Translation TTA (mirror + translate)
        # Pad by 1 pixel with reflection
        pad = 1
        padded_images = F.pad(images, (pad,)*4, 'reflect')
        
        # Two translations: top-left and bottom-right
        img_tl = padded_images[:, :, 0:32, 0:32]
        img_br = padded_images[:, :, 2:34, 2:34]
        
        # Mirror TTA for each translation
        logits_tl = 0.5 * (model(img_tl) + model(torch.flip(img_tl, [-1])))
        logits_br = 0.5 * (model(img_br) + model(torch.flip(img_br, [-1])))
        
        logits_translate = 0.5 * (logits_tl + logits_br)
        
        # Final TTA is 50% mirror(original) + 50% mirror(translated)
        avg_outputs = 0.5 * logits_mirror + 0.5 * logits_translate
        
    return outputs, avg_outputs

class EMA:
    def __init__(self, model, decay, ema_buffers=False):
        self.model = model
        self.decay = decay
        self.ema_buffers = ema_buffers
        self.shadow = {}
        self.backup = {}
        self.initialize_shadow()

    @torch.no_grad()
    def initialize_shadow(self):
        """Initialize shadow weights with current model state."""
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        
        if self.ema_buffers:
            for name, buffer in self.model.named_buffers():
                self.shadow[name] = buffer.data.clone().detach()

    @torch.no_grad()
    def update(self):
        """Update shadow weights with current model state."""
        # Update parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # shadow = shadow * decay + current * (1 - decay)
                self.shadow[name].lerp_(param.data.detach(), 1.0 - self.decay)
        
        # Update buffers if enabled
        if self.ema_buffers:
            for name, buffer in self.model.named_buffers():
                if buffer.is_floating_point():
                    self.shadow[name].lerp_(buffer.data.detach(), 1.0 - self.decay)
                else:
                    self.shadow[name].copy_(buffer.data.detach())

    @torch.no_grad()
    def apply_shadow(self):
        """Switch model to EMA weights using in-place copy."""
        if self.backup:
            return # Already applied
            
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name])
        
        if self.ema_buffers:
            for name, buffer in self.model.named_buffers():
                self.backup[name] = buffer.data.clone().detach()
                buffer.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        """Restore model to original weights."""
        if not self.backup:
            return # Nothing to restore
            
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        
        if self.ema_buffers:
            for name, buffer in self.model.named_buffers():
                if name in self.backup:
                    buffer.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Return the state of the EMA."""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
            'ema_buffers': self.ema_buffers
        }

    def load_state_dict(self, state_dict):
        """Load the EMA state and ensure it's on the correct device."""
        self.decay = state_dict['decay']
        self.ema_buffers = state_dict['ema_buffers']
        
        # Determine target device from model
        device = next(self.model.parameters()).device
        
        self.shadow = {}
        for name, tensor in state_dict['shadow'].items():
            self.shadow[name] = tensor.to(device)
