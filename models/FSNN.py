import torch
import torch.nn as nn
import torch.fft


class FSNNLayer(nn.Module):
    """
    Trainable FSNN Layer.
    
    """
    def __init__(self, channels, signal_len, alpha=2000.0):
        super(FSNNLayer, self).__init__()
        self.channels = channels
        self.signal_len = signal_len
        
        # 1. Make alpha a trainable parameter to allow dynamic frequency scaling
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

        # Trainable filters in frequency domain: (channels, channels, freq_bins)
        # Using complex float for the weights to model amplitude & phase shifts
        self.u_hat = nn.Parameter(torch.randn(channels, channels, signal_len // 2 + 1, dtype=torch.cfloat) * 1e-2)

        # Center frequencies for the filters (Deterministic linspace initialization)
        self.omegas = nn.Parameter(torch.linspace(0, 0.5, channels))

        # Register frequency bins as a buffer (computed once, stored on correct device automatically)
        self.register_buffer('freqs', torch.fft.rfftfreq(signal_len))

    def forward(self, x):
        # x shape: (Batch, Channels, Signal_Len)
        
        # 1. FFT
        f_hat = torch.fft.rfft(x, n=self.signal_len, dim=-1) # (B, C, F)

        # 2. Cross-Channel Mixing via Einsum
        # Mathematically identical to expanding tensors, but prevents O(B * C * C * F) OOM issues.
        mixed_spectra = torch.einsum('bif, ijf -> bjf', f_hat, self.u_hat)

        # 3. Wiener Filter Construction (Denominator)
        alpha_pos = torch.abs(self.alpha)
        denominator = 1.0 + 2.0 * alpha_pos * (self.freqs.unsqueeze(0) - self.omegas.unsqueeze(1))**2

        # 4. Apply Wiener Filter
        out_hat = mixed_spectra / denominator.unsqueeze(0)

        # 5. IFFT to return to time domain
        return torch.fft.irfft(out_hat, n=self.signal_len, dim=-1)


class FSNNBlock(nn.Module):
    """
    Standard Isotropic Block: FSNN -> GELU -> Dropout -> Residual -> LayerNorm
    """
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(FSNNBlock, self).__init__()
        self.FSNN = FSNNLayer(d_model, seq_len)
        
        # Reverted back to LayerNorm on the d_model dimension
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() 

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        
        # 1. Permute specifically for FSNN Layer
        inp = x.permute(0, 2, 1) # (Batch, d_model, Seq_Len)
        out = self.FSNN(inp)
        
        # 2. Permute back to standard isotropic shape
        out = out.permute(0, 2, 1) # (Batch, Seq_Len, d_model)
        
        # 3. Non-linearity & Dropout
        out = self.activation(out)
        out = self.dropout(out)
        
        # 4. Residual Connection & Layer Normalization
        return self.norm(x + out)


class Model(nn.Module):
    """
    Combined Deep Stacked FSNN Network.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        
        #  fetch d_model and e_layers, defaulting to 512 and 3 if missing
        d_model = getattr(configs, 'd_model', 512)
        e_layers = getattr(configs, 'e_layers', 3)
        dropout = getattr(configs, 'dropout', 0.1)
        
        # 1. Input Projection: Map raw input features up to d_model
        self.input_proj = nn.Linear(configs.enc_in, d_model)
        
        # 2. Encoder Stacking: Deep isotropic blocks with residuals
        self.encoder = nn.ModuleList([
            FSNNBlock(d_model, configs.seq_len, dropout)
            for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

        # 3. Task Specific Heads
        if self.task_name == 'classification':
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model, getattr(configs, 'num_class', 10))
            
        elif self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Project Channel Dimension: d_model -> c_out
            self.projection = nn.Linear(d_model, configs.c_out)
            # Project Temporal Dimension: seq_len -> pred_len (Allows P > L)
            self.time_proj = nn.Linear(configs.seq_len, self.pred_len)
            
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc shape: [Batch, Seq_Len, Enc_In]
        
        # 1. Project Input Features
        x = self.input_proj(x_enc) # [Batch, Seq_Len, d_model]
        
        # 2. Forward through FSNN Blocks (maintaining Batch, Seq_Len, d_model)
        for block in self.encoder:
            x = block(x)
            
        enc_out = self.norm(x)     # [Batch, Seq_Len, d_model]
        
        # 3. Apply appropriate Task Head
        if self.task_name == 'classification':
            # Permute for pooling over sequence length
            out = enc_out.permute(0, 2, 1)       # (Batch, d_model, Seq_Len)
            out = self.pool(out).squeeze(-1)     # (Batch, d_model)
            out = self.dropout(out)
            out = self.projection(out)           # (Batch, num_classes)
            return out
            
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Map Dimensions: d_model -> c_out
            out = self.projection(enc_out)       # (Batch, Seq_Len, c_out)
            
            # Map Time dimension
            out = out.permute(0, 2, 1)           # (Batch, c_out, Seq_Len)
            out = self.time_proj(out)            # (Batch, c_out, Pred_Len)
            
            # Final output shape: (Batch, Pred_Len, c_out)
            return out.permute(0, 2, 1)          
            
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            return self.projection(enc_out)      # (Batch, Seq_Len, c_out)
            
        return None