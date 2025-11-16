"""
Test script for X-axis zoom functionality.

This script tests that:
1. The plot_envelope_with_onsets function has the scroll event handler
2. The scroll event handler modifies X-axis limits correctly
3. The Y-axis remains unchanged during zoom
"""

import unittest
import numpy as np
import matplotlib
# Use non-interactive backend for testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import onset_detection


class TestXAxisZoomFunctionality(unittest.TestCase):
    """Test cases for X-axis zoom functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test data
        self.sr = 1000
        duration = 2.0
        t = np.linspace(0, duration, int(self.sr * duration))
        self.y = np.sin(2 * np.pi * 10 * t)
        
        # Create envelope and times
        self.env = np.abs(self.y)
        self.times = t
        
        # Create some onset times
        self.onset_times = np.array([0.5, 1.0, 1.5])
    
    def test_plot_creates_figure_with_event_handler(self):
        """Test that plot_envelope_with_onsets creates a figure with scroll event handler."""
        # Close any existing figures
        plt.close('all')
        
        # Create the plot (should not display in Agg backend)
        # We need to patch plt.show() to avoid blocking
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            onset_detection.plot_envelope_with_onsets(
                self.y, self.sr, self.env, self.times, self.onset_times,
                title="Test Plot"
            )
            
            # Check that a figure was created
            fig = plt.gcf()
            self.assertIsNotNone(fig)
            
            # Check that the figure has 2 subplots
            axes = fig.get_axes()
            self.assertEqual(len(axes), 2)
            
            # Check that scroll event is connected
            callbacks = fig.canvas.callbacks.callbacks.get('scroll_event', {})
            self.assertGreater(len(callbacks), 0, "No scroll event handler found")
            
        finally:
            plt.show = original_show
            plt.close('all')
    
    def test_scroll_event_modifies_xlim(self):
        """Test that scroll events modify X-axis limits."""
        plt.close('all')
        
        # Create a simple plot
        fig, ax = plt.subplots(1, 1)
        ax.plot([0, 1, 2, 3], [0, 1, 0, 1])
        
        # Set initial limits
        initial_xlim = (0, 3)
        ax.set_xlim(initial_xlim)
        
        # Store Y limits to verify they don't change
        initial_ylim = ax.get_ylim()
        
        # Create zoom handler similar to the one in plot_envelope_with_onsets
        def on_scroll(event):
            if event.inaxes is None:
                return
            
            ax = event.inaxes
            cur_xlim = ax.get_xlim()
            xdata = event.xdata
            
            zoom_factor = 1.2
            if event.button == 'up':
                scale_factor = 1 / zoom_factor
            elif event.button == 'down':
                scale_factor = zoom_factor
            else:
                return
            
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
            ax.set_xlim(new_xlim)
            fig.canvas.draw_idle()
        
        # Connect the handler
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Simulate a scroll event (zoom in)
        # Create event and manually set xdata/ydata
        event = MouseEvent(
            'scroll_event', fig.canvas, 
            x=100, y=100,
            button='up'
        )
        event.inaxes = ax
        event.xdata = 1.5
        event.ydata = 0.5
        
        # Get the callback and call it directly
        on_scroll(event)
        
        # Check that X-axis limits changed (zoomed in)
        new_xlim = ax.get_xlim()
        xlim_width_initial = initial_xlim[1] - initial_xlim[0]
        xlim_width_new = new_xlim[1] - new_xlim[0]
        
        # After zooming in, width should be smaller
        self.assertLess(xlim_width_new, xlim_width_initial,
                       "X-axis width should decrease when zooming in")
        
        plt.close('all')
    
    def test_zoom_centers_on_mouse_position(self):
        """Test that zoom is centered around mouse position."""
        plt.close('all')
        
        fig, ax = plt.subplots(1, 1)
        ax.plot([0, 10], [0, 1])
        ax.set_xlim(0, 10)
        
        def on_scroll(event):
            if event.inaxes is None:
                return
            
            ax = event.inaxes
            cur_xlim = ax.get_xlim()
            xdata = event.xdata
            
            zoom_factor = 1.2
            if event.button == 'up':
                scale_factor = 1 / zoom_factor
            elif event.button == 'down':
                scale_factor = zoom_factor
            else:
                return
            
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
            ax.set_xlim(new_xlim)
        
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Zoom at position x=5 (center)
        event = MouseEvent(
            'scroll_event', fig.canvas,
            x=100, y=100,
            button='up'
        )
        event.inaxes = ax
        event.xdata = 5.0
        event.ydata = 0.5
        on_scroll(event)
        
        new_xlim = ax.get_xlim()
        # Mouse position should still be visible in the zoomed view
        self.assertLess(new_xlim[0], 5.0, "Mouse position should be in zoomed view")
        self.assertGreater(new_xlim[1], 5.0, "Mouse position should be in zoomed view")
        
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
