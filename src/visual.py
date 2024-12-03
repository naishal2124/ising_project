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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, Button
from typing import Tuple, Dict, List, Set
from collections import deque
from tqdm import tqdm
import time

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
        """Plot current spin configuration with domain highlighting."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Raw spin configuration
        im1 = ax1.imshow(self.model.spins, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('Spin Configuration')
        plt.colorbar(im1, ax=ax1, label='Spin')

        # Domain visualization
        domains = self._identify_domains_iterative()
        im2 = ax2.imshow(domains, cmap='tab20')
        ax2.set_title('Domain Structure')
        plt.colorbar(im2, ax=ax2, label='Domain ID')

        if title:
            fig.suptitle(title)

        # Add measurements
        m = abs(self.model.magnetization/self.model.N)
        e = self.model.energy/self.model.N
        text = f'|m| = {m:.3f}\nE/N = {e:.3f}'
        fig.text(0.85, 0.85, text, transform=ax1.transAxes)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def initialize_system(self, T: float = None) -> None:
        """Initialize Ising model at given temperature."""
        if T is not None:
            self.T = T
        self.model = IsingModel(self.L, self.T)
        self.updater = ClusterUpdater(self.model)

        # Equilibrate system
        for _ in tqdm(range(self.n_equilibration), desc="Equilibrating"):
            self.updater.wolff_update()

    def _identify_domains_iterative(self) -> np.ndarray:
        """Identify connected domains using iterative approach."""
        L = self.model.L
        domains = np.zeros((L, L), dtype=int)
        domain_id = 1

        def get_domain(i: int, j: int, spin: int) -> Set[Tuple[int, int]]:
            """Get all points in a domain using BFS."""
            domain = set()
            queue = deque([(i, j)])

            while queue:
                current_i, current_j = queue.popleft()
                if (current_i < 0 or current_i >= L or
                    current_j < 0 or current_j >= L or
                    domains[current_i, current_j] != 0 or
                    self.model.spins[current_i, current_j] != spin):
                    continue

                domains[current_i, current_j] = domain_id
                domain.add((current_i, current_j))

                for ni, nj in self.model.get_neighbor_spins(current_i, current_j):
                    queue.append((ni, nj))

            return domain

        for i in range(L):
            for j in range(L):
                if domains[i, j] == 0:
                    domain = get_domain(i, j, self.model.spins[i, j])
                    if domain:  # Only increment domain_id if we found a domain
                        domain_id += 1

        return domains

    def _analyze_domains(self, domains: np.ndarray) -> Dict:
        """Analyze domain structure and statistics."""
        unique_domains = np.unique(domains[domains > 0])
        domain_sizes = [np.sum(domains == d) for d in unique_domains]

        L = self.model.L
        interface_length = 0
        for i in range(L):
            for j in range(L):
                current = domains[i, j]
                for ni, nj in self.model.get_neighbor_spins(i, j):
                    if domains[ni, nj] != current:
                        interface_length += 1

        return {
            'n_domains': len(unique_domains),
            'avg_size': np.mean(domain_sizes) if domain_sizes else 0,
            'max_size': max(domain_sizes) if domain_sizes else 0,
            'size_distribution': domain_sizes,
            'interface_length': interface_length // 2
        }

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

                self.im.set_array(self.model.spins)
                self.ax.set_title(f'T/Tc = {val:.3f}')
                self._update_measurements()
                self.fig.canvas.draw_idle()

        self.slider.on_changed(update)

        # Add play/pause button
        button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
        self.play_button = Button(button_ax, 'Play/Pause')
        self.play_button.on_clicked(self._play_pause)

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

    def create_evolution_animation(self, n_frames: int, 
                                   algorithm: str = 'wolff',
                                   save_path: str = None):
        """Create animation of system evolution."""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.model.spins, cmap='coolwarm', 
                      animated=True, vmin=-1, vmax=1)
        plt.colorbar(im)

        def update(frame):
            if algorithm == 'wolff':
                self.updater.wolff_update()
            else:
                self.updater.swendsen_wang_update()
            im.set_array(self.model.spins)
            return [im]

        anim = FuncAnimation(fig, update, frames=n_frames, 
                            interval=100, blit=True)

        if save_path:
            writer = PillowWriter(fps=10)
            anim.save(save_path, writer=writer)

        plt.show()

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

        # Setup and show interactive plot
        self.setup_plot()
        plt.show()

if __name__ == "__main__":
    # Test visualization
    print("Initializing model...")
    L = 50
    T = 2.27  # Critical temperature
    viz = IsingVisualizer(L)
    viz.show_evolution()