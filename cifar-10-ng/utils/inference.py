import torch
import torch.nn as nn

def predict(model, images, device):
    """Standard inference."""
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
    return outputs

def predict_tta(model, images, device):
    """Test-Time Augmentation: Original + Horizontal Flip."""
    model.eval()
    with torch.no_grad():
        # Original images
        outputs = model(images.to(device))
        # Horizontal flip: images shape (N, C, H, W), flip dimension 3
        outputs_flipped = model(torch.flip(images, [3]).to(device))
        avg_outputs = (outputs + outputs_flipped) / 2
    return avg_outputs

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
