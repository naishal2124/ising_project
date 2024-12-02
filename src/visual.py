import os
import sys
from pathlib import Path

# Add project root to Python path if needed
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Tuple, Dict
import time
from tqdm import tqdm
from src.ising import IsingModel
from src.cluster import ClusterUpdater

class IsingVisualizer:
    """Interactive visualization for 2D Ising model."""
    
    def __init__(self, L: int = 50):
        self.L = L
        self.T_critical = 2.269185
        self.model = None
        self.updater = None
        self.fig = None
        self.ax = None
        self.im = None
        self.slider = None
        self.T = self.T_critical
        self.running = False
        self.n_equilibration = 1000
    
    def plot_configuration(self, title: str = None, save_path: str = None):
        """Plot static configuration."""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(self.model.spins, cmap='coolwarm', vmin=-1, vmax=1)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'T/Tc = {self.T/self.T_critical:.3f}')
            
        plt.colorbar(im, label='Spin')
        
        # Add measurements
        m = abs(self.model.magnetization/self.model.N)
        e = self.model.energy/self.model.N
        text = f'|m| = {m:.3f}\nE/N = {e:.3f}'
        fig.text(0.85, 0.85, text, transform=ax.transAxes)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def initialize_system(self, T: float) -> None:
        """Initialize Ising model at given temperature."""
        self.T = T
        self.model = IsingModel(self.L, T)
        self.updater = ClusterUpdater(self.model)
        
        # Equilibrate system
        for _ in tqdm(range(self.n_equilibration), desc="Equilibrating"):
            self.updater.wolff_update()
    
    def setup_plot(self) -> None:
        """Setup the interactive plot with slider."""
        # Create figure with extra space for slider and text
        self.fig = plt.figure(figsize=(10, 12))
        self.ax = plt.subplot2grid((6, 1), (0, 0), rowspan=5)
        plt.subplots_adjust(bottom=0.2, right=0.85)
        
        # Initial plot
        self.im = self.ax.imshow(self.model.spins, cmap='coolwarm', vmin=-1, vmax=1)
        self.ax.set_title(f'T/Tc = {self.T/self.T_critical:.3f}')
        plt.colorbar(self.im, label='Spin')
        
        # Add measurements text box
        self.text_box = self.fig.add_axes([0.87, 0.7, 0.1, 0.2])
        self.text_box.axis('off')
        
        # Add slider
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(
            slider_ax, 'T/Tc',
            valmin=0.5, valmax=1.5,
            valinit=self.T/self.T_critical,
            valstep=0.01
        )
        
        def update(val):
            new_T = val * self.T_critical
            if abs(new_T - self.T) > 1e-6:
                self.T = new_T
                self.model = IsingModel(self.L, self.T)
                self.updater = ClusterUpdater(self.model)
                
                for _ in range(100):
                    self.updater.wolff_update()
                
                self.im.set_array(self.model.spins)
                self.ax.set_title(f'T/Tc = {val:.3f}')
                self._update_measurements()
                self.fig.canvas.draw_idle()
        
        self.slider.on_changed(update)
        
        # Add play/pause button
        button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.play_button = Button(button_ax, 'Play/Pause')
        self.play_button.on_clicked(self._play_pause)
        
        self._update_measurements()
    
    def _update_measurements(self):
        """Update measurements display."""
        m = abs(self.model.magnetization/self.model.N)
        e = self.model.energy/self.model.N
        text = f'Measurements:\n\n|m| = {m:.3f}\n\nE/N = {e:.3f}'
        
        # Clear previous text
        self.text_box.clear()
        self.text_box.text(0.0, 0.9, text, verticalalignment='top')
    
    def _play_pause(self, event):
        """Handle play/pause button click."""
        self.running = not self.running
        if self.running:
            self.animate()
    
    def animate(self):
        """Animate system evolution."""
        while self.running:
            self.updater.wolff_update()
            self.im.set_array(self.model.spins)
            self._update_measurements()
            plt.pause(0.1)
            if not plt.fignum_exists(self.fig.number):
                break
    
    def demonstrate_phases(self):
        """Show configurations at different temperatures."""
        T_values = [0.5 * self.T_critical, self.T_critical, 1.5 * self.T_critical]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for T, ax in zip(T_values, axes):
            self.initialize_system(T)
            im = ax.imshow(self.model.spins, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(f'T/Tc = {T/self.T_critical:.3f}')
            plt.colorbar(im, ax=ax, label='Spin')
            
            m = abs(self.model.magnetization/self.model.N)
            e = self.model.energy/self.model.N
            ax.text(1.05, 0.95, f'|m| = {m:.3f}\nE/N = {e:.3f}', 
                   transform=ax.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    def show_evolution(self):
        """Show interactive visualization."""
        # Initialize at critical temperature
        self.initialize_system(self.T_critical)
        
        # Plot initial configuration
        self.plot_configuration("Initial Configuration")
        
        # Setup and show interactive plot
        self.setup_plot()
        plt.show()

def create_interactive_visualization(L: int = 50):
    """Create and show interactive visualization."""
    viz = IsingVisualizer(L)
    
    # Show different phases
    print("Demonstrating different phases...")
    viz.demonstrate_phases()
    
    # Show interactive visualization
    print("\nStarting interactive visualization...")
    viz.show_evolution()