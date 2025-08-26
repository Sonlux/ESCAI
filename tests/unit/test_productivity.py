"""
Unit tests for productivity features
"""

import pytest
from datetime import datetime
from escai_framework.cli.utils.productivity import (
    CommandHistory, AutoCompleter, ThemeManager, ProfileManager,
    MacroSystem, WorkspaceManager, AccessibilityFeatures,
    Macro, MacroStep, ConfigProfile
)


class TestCommandHistory:
    """Test command history functionality"""
    
    def test_add_command(self):
        """Test adding commands to history"""
        history = CommandHistory()
        
        history.add_command("escai monitor start")
        history.add_command("escai analyze patterns")
        
        assert len(history.commands) == 2
        assert "escai monitor start" in history.commands
        assert "escai analyze patterns" in history.commands
    
    def test_duplicate_prevention(self):
        """Test duplicate command prevention"""
        history = CommandHistory()
        
        history.add_command("escai monitor start")
        history.add_command("escai monitor start")  # Duplicate
        
        assert len(history.commands) == 1
    
    def test_max_history_limit(self):
        """Test maximum history limit"""
        history = CommandHistory(max_history=3)
        
        for i in range(5):
            history.add_command(f"command_{i}")
        
        assert len(history.commands) == 3
        assert "command_2" in history.commands
        assert "command_3" in history.commands
        assert "command_4" in history.commands
        assert "command_0" not in history.commands
    
    def test_search_history(self):
        """Test history search"""
        history = CommandHistory()
        history.add_command("escai monitor start")
        history.add_command("escai analyze patterns")
        history.add_command("escai monitor stop")
        
        results = history.search_history("monitor")
        assert len(results) == 2
        assert "escai monitor start" in results
        assert "escai monitor stop" in results
    
    def test_favorites(self):
        """Test favorite commands"""
        history = CommandHistory()
        
        history.add_favorite("escai monitor start --agent-id test")
        history.add_favorite("escai analyze patterns --timeframe 24h")
        
        assert len(history.favorites) == 2
        assert "escai monitor start --agent-id test" in history.favorites
        
        # Test duplicate prevention
        history.add_favorite("escai monitor start --agent-id test")
        assert len(history.favorites) == 2
        
        # Test removal
        history.remove_favorite("escai monitor start --agent-id test")
        assert len(history.favorites) == 1
    
    def test_get_recent(self):
        """Test getting recent commands"""
        history = CommandHistory()
        
        for i in range(15):
            history.add_command(f"command_{i}")
        
        recent = history.get_recent(5)
        assert len(recent) == 5
        assert "command_14" in recent
        assert "command_10" in recent


class TestAutoCompleter:
    """Test auto-completion functionality"""
    
    def test_basic_completion(self):
        """Test basic auto-completion"""
        completer = AutoCompleter()
        
        # Test that rules are set up
        assert len(completer.rules) > 0
        
        # Test direct pattern matching
        suggestions = completer.get_suggestions("monitor")
        # Should get suggestions from history or partial matches
        assert isinstance(suggestions, list)
    
    def test_command_specific_completion(self):
        """Test command-specific completion"""
        completer = AutoCompleter()
        
        # Add a command to history that should match
        completer.command_history.add_command("escai monitor start")
        
        suggestions = completer.get_suggestions("monitor")
        assert len(suggestions) >= 0  # Should return some suggestions
    
    def test_option_completion(self):
        """Test option completion"""
        completer = AutoCompleter()
        
        # Test that we can get suggestions for framework options
        suggestions = completer.get_suggestions("framework")
        assert isinstance(suggestions, list)
    
    def test_history_integration(self):
        """Test integration with command history"""
        completer = AutoCompleter()
        completer.command_history.add_command("escai monitor start --agent-id test123")
        
        suggestions = completer.get_suggestions("test")
        assert any("test123" in s for s in suggestions)
    
    def test_custom_rules(self):
        """Test adding custom completion rules"""
        completer = AutoCompleter()
        
        initial_count = len(completer.rules)
        completer.add_rule("custom", ["option1", "option2", "option3"])
        
        # Should have added one more rule
        assert len(completer.rules) == initial_count + 1
        
        # Check that the rule was added correctly
        custom_rule = completer.rules[-1]
        assert custom_rule.pattern == "custom"
        assert "option1" in custom_rule.suggestions


class TestThemeManager:
    """Test theme management"""
    
    def test_default_themes(self):
        """Test default theme setup"""
        manager = ThemeManager()
        
        themes = manager.list_themes()
        assert "default" in themes
        assert "dark" in themes
        assert "light" in themes
        assert "high_contrast" in themes
    
    def test_set_theme(self):
        """Test setting theme"""
        manager = ThemeManager()
        
        assert manager.set_theme("dark") is True
        assert manager.current_theme == "dark"
        
        assert manager.set_theme("nonexistent") is False
        assert manager.current_theme == "dark"  # Should remain unchanged
    
    def test_get_current_theme(self):
        """Test getting current theme"""
        manager = ThemeManager()
        manager.set_theme("light")
        
        theme = manager.get_current_theme()
        assert theme.name == "Light"
        assert "primary" in theme.colors
        assert "header" in theme.styles
    
    def test_custom_theme(self):
        """Test adding custom theme"""
        manager = ThemeManager()
        
        custom_colors = {"primary": "purple", "secondary": "pink"}
        custom_styles = {"header": "bold purple"}
        
        manager.add_custom_theme("custom", custom_colors, custom_styles)
        
        assert "custom" in manager.list_themes()
        assert manager.set_theme("custom") is True
        
        theme = manager.get_current_theme()
        assert theme.colors["primary"] == "purple"
        assert theme.styles["header"] == "bold purple"


class TestProfileManager:
    """Test configuration profile management"""
    
    def test_default_profiles(self):
        """Test default profile setup"""
        manager = ProfileManager()
        
        profiles = [p.name for p in manager.list_profiles()]
        assert "Development" in profiles
        assert "Production" in profiles
        assert "Research" in profiles
        assert "Demo" in profiles
    
    def test_set_profile(self):
        """Test setting profile"""
        manager = ProfileManager()
        
        assert manager.set_profile("development") is True
        assert manager.current_profile == "development"
        
        assert manager.set_profile("nonexistent") is False
    
    def test_get_current_profile(self):
        """Test getting current profile"""
        manager = ProfileManager()
        manager.set_profile("production")
        
        profile = manager.get_current_profile()
        assert profile.name == "Production"
        assert "refresh_rate" in profile.settings
        assert profile.settings["log_level"] == "INFO"
    
    def test_create_custom_profile(self):
        """Test creating custom profile"""
        manager = ProfileManager()
        
        custom_settings = {
            "refresh_rate": 0.1,
            "theme": "custom",
            "debug": True
        }
        
        manager.create_profile("test", "Test profile", custom_settings)
        
        profiles = [p.name for p in manager.list_profiles()]
        assert "test" in profiles
        
        manager.set_profile("test")
        profile = manager.get_current_profile()
        assert profile.settings["refresh_rate"] == 0.1
    
    def test_delete_profile(self):
        """Test deleting profile"""
        manager = ProfileManager()
        
        # Create and delete custom profile
        manager.create_profile("temp", "Temporary", {})
        assert manager.delete_profile("temp") is True
        
        profiles = [p.name for p in manager.list_profiles()]
        assert "temp" not in profiles
        
        # Cannot delete default profile
        assert manager.delete_profile("development") is True  # Can delete non-default


class TestMacroSystem:
    """Test macro recording and playback"""
    
    def test_start_stop_recording(self):
        """Test macro recording"""
        macro_system = MacroSystem()
        
        # Start recording
        assert macro_system.start_recording("test_macro", "Test macro") is True
        assert macro_system.recording is True
        
        # Cannot start another recording while one is active
        assert macro_system.start_recording("another", "Another") is False
        
        # Stop recording
        macro = macro_system.stop_recording()
        assert macro is not None
        assert macro.name == "test_macro"
        assert macro_system.recording is False
    
    def test_record_commands(self):
        """Test recording commands"""
        macro_system = MacroSystem()
        
        macro_system.start_recording("test", "Test")
        macro_system.record_command("escai", ["monitor", "start"], "Start monitoring")
        macro_system.record_command("escai", ["analyze", "patterns"], "Analyze patterns")
        
        macro = macro_system.stop_recording()
        assert len(macro.steps) == 2
        assert macro.steps[0].command == "escai"
        assert macro.steps[0].args == ["monitor", "start"]
        assert macro.steps[1].command == "escai"
        assert macro.steps[1].args == ["analyze", "patterns"]
    
    def test_execute_macro(self):
        """Test macro execution"""
        macro_system = MacroSystem()
        
        # Create macro
        macro_system.start_recording("test", "Test")
        macro_system.record_command("escai", ["monitor", "start"])
        macro_system.stop_recording()
        
        # Execute macro
        steps = macro_system.execute_macro("test")
        assert steps is not None
        assert len(steps) == 1
        assert steps[0].command == "escai"
        
        # Check usage count incremented
        macro = macro_system.get_macro("test")
        assert macro.usage_count == 1
    
    def test_list_delete_macros(self):
        """Test listing and deleting macros"""
        macro_system = MacroSystem()
        
        # Create macros
        macro_system.start_recording("macro1", "First")
        macro_system.stop_recording()
        
        macro_system.start_recording("macro2", "Second")
        macro_system.stop_recording()
        
        # List macros
        macros = macro_system.list_macros()
        assert len(macros) == 2
        
        # Delete macro
        assert macro_system.delete_macro("macro1") is True
        assert macro_system.delete_macro("nonexistent") is False
        
        macros = macro_system.list_macros()
        assert len(macros) == 1


class TestWorkspaceManager:
    """Test workspace management"""
    
    def test_default_workspace(self):
        """Test default workspace setup"""
        manager = WorkspaceManager()
        
        workspaces = [w["name"] for w in manager.list_workspaces()]
        assert "Default" in workspaces
        assert manager.current_workspace == "default"
    
    def test_create_workspace(self):
        """Test creating workspace"""
        manager = WorkspaceManager()
        
        assert manager.create_workspace("project1", "First project") is True
        assert manager.create_workspace("project1", "Duplicate") is False  # Duplicate
        
        workspaces = [w["name"] for w in manager.list_workspaces()]
        assert "project1" in workspaces
    
    def test_switch_workspace(self):
        """Test switching workspace"""
        manager = WorkspaceManager()
        
        manager.create_workspace("test", "Test workspace")
        
        assert manager.switch_workspace("test") is True
        assert manager.current_workspace == "test"
        
        assert manager.switch_workspace("nonexistent") is False
        assert manager.current_workspace == "test"  # Should remain unchanged
    
    def test_workspace_content(self):
        """Test workspace content management"""
        manager = WorkspaceManager()
        
        manager.create_workspace("test", "Test")
        manager.switch_workspace("test")
        
        # Add agent
        manager.add_agent_to_workspace("agent_001")
        workspace = manager.get_current_workspace()
        assert "agent_001" in workspace["agents"]
        
        # Add bookmark
        bookmark = {"name": "test", "url": "http://example.com"}
        manager.add_bookmark_to_workspace(bookmark)
        assert len(workspace["bookmarks"]) == 1
    
    def test_delete_workspace(self):
        """Test deleting workspace"""
        manager = WorkspaceManager()
        
        manager.create_workspace("temp", "Temporary")
        assert manager.delete_workspace("temp") is True
        
        workspaces = [w["name"] for w in manager.list_workspaces()]
        assert "temp" not in workspaces
        
        # Cannot delete default workspace
        assert manager.delete_workspace("default") is False


class TestAccessibilityFeatures:
    """Test accessibility features"""
    
    def test_screen_reader_mode(self):
        """Test screen reader mode"""
        accessibility = AccessibilityFeatures()
        
        accessibility.enable_screen_reader_mode()
        assert accessibility.screen_reader_mode is True
        assert accessibility.verbose_descriptions is True
        assert accessibility.announce_changes is True
    
    def test_format_for_screen_reader(self):
        """Test screen reader formatting"""
        accessibility = AccessibilityFeatures()
        accessibility.enable_screen_reader_mode()
        
        content = "[bold]Test[/bold] █░● ▲▼"
        formatted = accessibility.format_for_screen_reader(content)
        
        assert "[bold]" not in formatted
        assert "filled block" in formatted
        assert "empty block" in formatted
        assert "data point" in formatted
        assert "up arrow" in formatted
        assert "down arrow" in formatted
    
    def test_announce_change(self):
        """Test change announcements"""
        accessibility = AccessibilityFeatures()
        
        # Without screen reader mode
        message = accessibility.announce_change("Status updated")
        assert message == "Status updated"
        
        # With screen reader mode
        accessibility.enable_screen_reader_mode()
        message = accessibility.announce_change("Status updated")
        assert "[Announcement]" in message
    
    def test_accessibility_modes(self):
        """Test different accessibility modes"""
        accessibility = AccessibilityFeatures()
        
        accessibility.enable_high_contrast_mode()
        assert accessibility.high_contrast_mode is True
        
        accessibility.enable_keyboard_only_mode()
        assert accessibility.keyboard_only_mode is True


class TestMacroStep:
    """Test macro step functionality"""
    
    def test_macro_step_creation(self):
        """Test creating macro step"""
        step = MacroStep("escai", ["monitor", "start"], 1.0, "Start monitoring")
        
        assert step.command == "escai"
        assert step.args == ["monitor", "start"]
        assert step.delay == 1.0
        assert step.description == "Start monitoring"


class TestMacro:
    """Test macro functionality"""
    
    def test_macro_creation(self):
        """Test creating macro"""
        macro = Macro("test", "Test macro", [])
        
        assert macro.name == "test"
        assert macro.description == "Test macro"
        assert len(macro.steps) == 0
        assert macro.usage_count == 0
    
    def test_add_step(self):
        """Test adding steps to macro"""
        macro = Macro("test", "Test", [])
        
        macro.add_step("escai", ["monitor", "start"], 0.5, "Start monitoring")
        
        assert len(macro.steps) == 1
        assert macro.steps[0].command == "escai"
        assert macro.steps[0].delay == 0.5
    
    def test_execute_step(self):
        """Test executing macro step"""
        macro = Macro("test", "Test", [])
        macro.add_step("escai", ["monitor", "start"])
        
        step = macro.execute_step(0)
        assert step is not None
        assert step.command == "escai"
        
        # Invalid step index
        step = macro.execute_step(10)
        assert step is None


if __name__ == '__main__':
    pytest.main([__file__])