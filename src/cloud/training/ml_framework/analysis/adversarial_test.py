"""
Adversarial Testing - Model Vulnerability Detection

Tests model robustness against adversarial attacks and perturbations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class AdversarialTester:
    """
    Adversarial testing for model vulnerability detection.
    
    As noted in deep learning literature, models can be easily fooled.
    This module tests model robustness.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize adversarial tester.
        
        Args:
            model: Neural network model
            device: Device to run tests on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("adversarial_tester_initialized")
    
    def fgsm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            x: Input tensor
            y: Target tensor
            epsilon: Perturbation magnitude
            
        Returns:
            Adversarial example
        """
        x.requires_grad = True
        
        # Forward pass
        output = self.model(x)
        loss = F.mse_loss(output, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        x_grad = x.grad.data
        x_adv = x + epsilon * x_grad.sign()
        
        return x_adv.detach()
    
    def pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iter: int = 10,
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack.
        
        Args:
            x: Input tensor
            y: Target tensor
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations
            
        Returns:
            Adversarial example
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(num_iter):
            output = self.model(x_adv)
            loss = F.mse_loss(output, y)
            
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                # Clip to epsilon ball
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
                x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv.detach()
    
    def test_robustness(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Test model robustness against adversarial attacks.
        
        Args:
            x: Input tensor
            y: Target tensor
            attack_type: Type of attack ("fgsm", "pgd")
            epsilon: Perturbation magnitude
            
        Returns:
            Dictionary with robustness metrics
        """
        # Original predictions
        with torch.no_grad():
            output_original = self.model(x)
            loss_original = F.mse_loss(output_original, y).item()
        
        # Generate adversarial example
        if attack_type == "fgsm":
            x_adv = self.fgsm_attack(x, y, epsilon)
        elif attack_type == "pgd":
            x_adv = self.pgd_attack(x, y, epsilon)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Adversarial predictions
        with torch.no_grad():
            output_adv = self.model(x_adv)
            loss_adv = F.mse_loss(output_adv, y).item()
        
        # Calculate robustness metrics
        perturbation_norm = torch.norm(x_adv - x).item()
        prediction_change = torch.norm(output_adv - output_original).item()
        
        # Vulnerability score (higher = more vulnerable)
        vulnerability_score = (loss_adv - loss_original) / loss_original if loss_original > 0 else 0.0
        
        return {
            "attack_type": attack_type,
            "epsilon": epsilon,
            "loss_original": loss_original,
            "loss_adv": loss_adv,
            "perturbation_norm": perturbation_norm,
            "prediction_change": prediction_change,
            "vulnerability_score": vulnerability_score,
            "is_vulnerable": vulnerability_score > 0.1,  # Threshold
        }
    
    def test_noise_robustness(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
    ) -> Dict[str, Any]:
        """
        Test model robustness against random noise.
        
        Args:
            x: Input tensor
            y: Target tensor
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary with robustness metrics for each noise level
        """
        results = {}
        
        with torch.no_grad():
            output_original = self.model(x)
            loss_original = F.mse_loss(output_original, y).item()
        
        for noise_level in noise_levels:
            # Add noise
            noise = torch.randn_like(x) * noise_level
            x_noisy = x + noise
            
            # Predict
            with torch.no_grad():
                output_noisy = self.model(x_noisy)
                loss_noisy = F.mse_loss(output_noisy, y).item()
            
            results[f"noise_{noise_level}"] = {
                "loss_original": loss_original,
                "loss_noisy": loss_noisy,
                "degradation": (loss_noisy - loss_original) / loss_original if loss_original > 0 else 0.0,
            }
        
        return results
    
    def generate_adversarial_dataset(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """
        Generate adversarial dataset.
        
        Args:
            x: Input tensor
            y: Target tensor
            attack_type: Type of attack
            epsilon: Perturbation magnitude
            
        Returns:
            Adversarial examples
        """
        if attack_type == "fgsm":
            x_adv = self.fgsm_attack(x, y, epsilon)
        elif attack_type == "pgd":
            x_adv = self.pgd_attack(x, y, epsilon)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return x_adv

