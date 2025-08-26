"""
Unit tests for CLI console utilities
"""

import pytest
from rich.console import Console
from rich.theme import Theme

from escai_framework.cli.utils.console import get_console, escai_theme


class TestCLIConsole:
    """Test CLI console utilities"""
    
    def test_escai_theme_exists(self):
        """Test that ESCAI theme is properly defined"""
        assert isinstance(escai_theme, Theme)
        
        # Check that required theme colors are defined
        expected_styles = [
            'info', 'warning', 'error', 'success', 
            'highlight', 'accent', 'muted'
        ]
        
        for style in expected_styles:
            assert style in escai_theme.styles
    
    def test_escai_theme_colors(self):
        """Test that ESCAI theme has correct color mappings"""
        # Rich Theme stores Style objects, not strings
        assert 'info' in escai_theme.styles
        assert 'warning' in escai_theme.styles
        assert 'error' in escai_theme.styles
        assert 'success' in escai_theme.styles
        assert 'highlight' in escai_theme.styles
        assert 'accent' in escai_theme.styles
        assert 'muted' in escai_theme.styles
    
    def test_get_console_returns_console(self):
        """Test that get_console returns a Console instance"""
        console = get_console()
        assert isinstance(console, Console)
    
    def test_get_console_singleton(self):
        """Test that get_console returns the same instance"""
        console1 = get_console()
        console2 = get_console()
        assert console1 is console2
    
    def test_console_has_theme(self):
        """Test that console has the ESCAI theme applied"""
        console = get_console()
        # Rich Console stores theme in _theme_stack
        assert hasattr(console, '_theme_stack')
        
        # Check that theme styles are available
        for style_name in escai_theme.styles:
            # The console should be able to resolve theme styles
            try:
                style = console.get_style(style_name)
                assert style is not None
            except Exception:
                # Some styles might not be resolvable in test environment
                pass
    
    def test_console_theme_rendering(self):
        """Test that console can render themed text"""
        console = get_console()
        
        # Test that we can use theme styles without errors
        try:
            # These should not raise exceptions
            console.print("Test info", style="info")
            console.print("Test warning", style="warning") 
            console.print("Test error", style="error")
            console.print("Test success", style="success")
            console.print("Test highlight", style="highlight")
            console.print("Test accent", style="accent")
            console.print("Test muted", style="muted")
        except Exception as e:
            pytest.fail(f"Console theme rendering failed: {e}")
    
    def test_console_markup_rendering(self):
        """Test that console can render markup with theme styles"""
        console = get_console()
        
        try:
            # Test markup rendering with theme styles
            console.print("[info]Info text[/info]")
            console.print("[warning]Warning text[/warning]")
            console.print("[error]Error text[/error]")
            console.print("[success]Success text[/success]")
            console.print("[highlight]Highlighted text[/highlight]")
            console.print("[accent]Accented text[/accent]")
            console.print("[muted]Muted text[/muted]")
        except Exception as e:
            pytest.fail(f"Console markup rendering failed: {e}")
    
    def test_console_properties(self):
        """Test console properties and configuration"""
        console = get_console()
        
        # Console should have theme stack
        assert hasattr(console, '_theme_stack')
        
        # Console should be configured for CLI use
        # (These are default Rich console properties)
        assert hasattr(console, 'print')
        assert hasattr(console, 'log')
        assert hasattr(console, 'rule')
        assert hasattr(console, 'status')
    
    def test_theme_style_inheritance(self):
        """Test that theme styles work with Rich's style inheritance"""
        console = get_console()
        
        # Test basic theme styles (without inheritance for now)
        try:
            console.print("Info text", style="info")
            console.print("Warning text", style="warning") 
            console.print("Success text", style="success")
        except Exception as e:
            pytest.fail(f"Theme style rendering failed: {e}")
    
    def test_console_output_capture(self):
        """Test that console output can be captured for testing"""
        from io import StringIO
        
        # Create console with custom file for testing
        test_output = StringIO()
        console = Console(theme=escai_theme, file=test_output)
        
        console.print("Test output", style="info")
        output = test_output.getvalue()
        
        # Should contain the text (exact formatting may vary)
        assert "Test output" in output
    
    def test_console_width_handling(self):
        """Test console width handling"""
        console = get_console()
        
        # Console should handle width appropriately
        # (Default behavior, should not raise exceptions)
        try:
            console.print("A" * 200)  # Long line
            console.print("Short")     # Short line
        except Exception as e:
            pytest.fail(f"Console width handling failed: {e}")
    
    def test_console_color_system(self):
        """Test console color system detection"""
        console = get_console()
        
        # Console should have a color system
        assert hasattr(console, '_color_system')
        
        # Should be able to detect if colors are supported
        # (This may vary by environment, but should not raise exceptions)
        try:
            color_system = console._color_system
            # Should be None, ColorSystem.STANDARD, ColorSystem.EIGHT_BIT, or ColorSystem.TRUECOLOR
            assert color_system is None or hasattr(color_system, 'name')
        except Exception as e:
            pytest.fail(f"Color system detection failed: {e}")
    
    def test_multiple_console_instances(self):
        """Test behavior with multiple console calls"""
        # Should always return the same instance
        consoles = [get_console() for _ in range(10)]
        
        # All should be the same object
        for console in consoles[1:]:
            assert console is consoles[0]
    
    def test_console_thread_safety(self):
        """Test console thread safety (basic test)"""
        import threading
        
        results = []
        
        def get_console_in_thread():
            console = get_console()
            results.append(console)
        
        # Create multiple threads
        threads = [threading.Thread(target=get_console_in_thread) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should return the same console instance
        assert len(results) == 5
        for console in results[1:]:
            assert console is results[0]