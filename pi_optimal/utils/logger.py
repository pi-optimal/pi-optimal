# logger.py
import logging
from datetime import datetime
from typing import Optional, Dict, Set
import sys
from tqdm.auto import tqdm

class Logger:
    """
    Adaptive logger that automatically switches between notebook and terminal formatting.
    Logs are always displayed without toggle functionality, emojis render as actual emojis,
    support indentation levels for hierarchical logging, include visual connectors
    between indent levels in notebooks, and omit timestamp and log level in notebook display.
    """
    
    # Class-level dictionary to hold display handles per logger name
    _display_handles: Dict[str, 'IPython.display.DisplayHandle'] = {}
    
    # Class-level set to track which loggers have applied CSS
    _css_applied: Set[str] = set()
    
    EMOJI_MAP = {
        'DEBUG': '🔍',
        'INFO': '📝',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨',
        'PROCESS': '⚙️',
        'LOAD': '📂',
        'SAVE': '💾',
        'SUCCESS': '✨',
        'START': '🚀',
        'DONE': '🎉',
        'CLEAN': '🧹',
        'STATS': '📊',
        'CHECK': '✅',
    }

    COLOR_MAP = {
        'DEBUG': '#6c757d',    # Gray
        'INFO': '#0d6efd',     # Blue
        'WARNING': '#ffc107',  # Yellow
        'ERROR': '#dc3545',    # Red
        'CRITICAL': '#7b1fa2'  # Purple
    }

    MAX_LOG_MESSAGES = 100  # Optional: Limit the number of stored log messages

    INDENT_STEP = 20  # Pixels per indentation level

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize logger with automatic environment detection.

        Args:
            name: Name for the logger.
            level: Logging level (default: INFO).
        """
        self.name = name
        self.level = level
        self.log_messages = []

        self._setup_logger()

        if self._is_notebook():
            self._initialize_display_area()

    def _is_notebook(self) -> bool:
        """Check if code is running in a Jupyter notebook."""
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
                return True
            elif shell == 'TerminalInteractiveShell':  # Terminal IPython
                return False
            else:
                return False
        except NameError:  # Regular Python interpreter
            return False

    def _setup_logger(self):
        """Configure logger based on the detected environment."""
        if self._is_notebook():
            # Notebook environment - use HTML display
            try:
                from IPython.display import display, HTML, DisplayHandle
                self.display_handle = Logger._display_handles.get(self.name)
                if self.display_handle is None:
                    # Create a new display area if not already present
                    initial_html = self._generate_initial_html()
                    self.display_handle = display(HTML(initial_html), display_id=True)
                    Logger._display_handles[self.name] = self.display_handle
                self._display_func = self._notebook_display

                # Apply CSS only once per logger name
                if self.name not in Logger._css_applied:
                    self._apply_css()
                    Logger._css_applied.add(self.name)

            except ImportError:
                # Fallback to terminal logging if IPython isn't available
                self._setup_terminal_logger()
                self._display_func = self._terminal_display
        else:
            # Terminal environment - use standard logging
            self._setup_terminal_logger()
            self._display_func = self._terminal_display

    def _setup_terminal_logger(self):
        """Setup traditional logging for terminal environment using tqdm.write."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers to prevent duplicate logs
        self.logger.handlers.clear()
        
        # Create a custom handler that uses tqdm.write
        handler = TqdmLoggingHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _terminal_display(self, level: str, msg: str, emoji_key: Optional[str] = None, indent_level: int = 0):
        """Display log message in terminal using tqdm.write with indentation."""
        from tqdm import tqdm  # Import here to avoid issues if tqdm is not installed

        emoji = self.EMOJI_MAP.get(emoji_key, '') if emoji_key else ''
        formatted_msg = f"{emoji} {msg}" if emoji else msg
        indent_spaces = ' ' * (4 * indent_level)  # 4 spaces per indent level
        formatted_msg = f"{indent_spaces}{formatted_msg}"
        
        # Use the appropriate logging method
        if level == 'DEBUG':
            self.logger.debug(formatted_msg)
        elif level == 'INFO':
            self.logger.info(formatted_msg)
        elif level == 'WARNING':
            self.logger.warning(formatted_msg)
        elif level == 'ERROR':
            self.logger.error(formatted_msg)
        elif level == 'CRITICAL':
            self.logger.critical(formatted_msg)


    def _initialize_display_area(self):
        """Create a display area for grouped logs in the notebook."""
        # This method is no longer needed since display_handle is managed in _setup_logger
        pass

    def _generate_initial_html(self) -> str:
        """Generate the initial HTML structure without CSS."""
        return """
        <div id='logger-container'>
            <!-- Log messages will appear here -->
        </div>
        """

    def _apply_css(self):
        """Apply CSS styles to the logger container."""
        from IPython.display import display, HTML
        css = """
        <style>
            /* Logger Container */
            #logger-container {
                max-height: 600px;
                overflow-y: auto;
                font-family: 'Segoe UI Emoji', 'Segoe UI Symbol', monospace;
                padding: 5px;
                background-color: #ffffff;
                font-size: 0.9em; /* Increased font size */
                position: relative;
            }

            /* Individual Log Entry */
            .logger-entry {
                display: flex;
                align-items: center;
                padding: 1px 5px;
                margin-bottom: 4px;
                background-color: #ffffff;
                transition: background-color 0.2s, box-shadow 0.2s;
                white-space: pre-wrap; /* Preserve whitespace for symbols */
                position: relative;
            }

            .logger-entry:hover {
                background-color: #f1f3f5;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }

        

            /* Connector Line styling based on indent */
            .logger-entry[data-indent="1"]::before {
                left: 0;
            }
            .logger-entry[data-indent="2"]::before {
                left: 0;
            }
            /* Add more as needed for higher indent levels */

            /* Emoji */
            .logger-emoji {
                margin-right: 10px;
                font-size: 1em; /* Larger emoji size */
                flex-shrink: 0;
                color: inherit; /* Inherit color from parent */
            }

            /* Message */
            .logger-message {
                flex-grow: 1;
                color: #212529;
                word-break: break-word;
            }
        </style>
        """
        display(HTML(css))

    def _notebook_display(self, level: str, msg: str, emoji_key: Optional[str] = None, indent_level: int = 0):
        """Append log message to the display area in the notebook with indentation and visual connectors."""
        from IPython.display import HTML

        self.log_messages.append((level, msg, emoji_key, indent_level))
        
        # Optional: Limit the number of stored log messages
        if len(self.log_messages) > self.MAX_LOG_MESSAGES:
            self.log_messages.pop(0)
        
        # Generate HTML for all log messages
        html_logs = ""
        for idx, (lvl, message, emoji_key, indent) in enumerate(self.log_messages):
            emoji = self.EMOJI_MAP.get(emoji_key, '') if emoji_key else ''
            color = self.COLOR_MAP.get(lvl, '#000000')
            # Calculate indentation in pixels (e.g., 20px per indent level)
            indent_px = self.INDENT_STEP * indent
            html_logs += f"""
            <div class="logger-entry" data-indent="{indent}" style="margin-left: {indent_px}px;">
                <span class="logger-emoji" style="color: {color};">{emoji}</span>
                <span class="logger-message">{message}</span>
            </div>
            """

        # Update the display with the new HTML content
        if self.display_handle:
            self.display_handle.update(HTML(f"<div id='logger-container'>{html_logs}</div>"))

    def debug(self, msg: str, emoji_key: Optional[str] = 'DEBUG', indent_level: int = 0):
        """
        Log a debug message.

        Args:
            msg: The message to log.
            emoji_key: The key to retrieve the corresponding emoji.
            indent_level: The indentation level for the message.
        """
        self._display_func('DEBUG', msg, emoji_key, indent_level)

    def info(self, msg: str, emoji_key: Optional[str] = 'INFO', indent_level: int = 0):
        """
        Log an info message.

        Args:
            msg: The message to log.
            emoji_key: The key to retrieve the corresponding emoji.
            indent_level: The indentation level for the message.
        """
        self._display_func('INFO', msg, emoji_key, indent_level)

    def warning(self, msg: str, emoji_key: Optional[str] = 'WARNING', indent_level: int = 0):
        """
        Log a warning message.

        Args:
            msg: The message to log.
            emoji_key: The key to retrieve the corresponding emoji.
            indent_level: The indentation level for the message.
        """
        self._display_func('WARNING', msg, emoji_key, indent_level)

    def error(self, msg: str, emoji_key: Optional[str] = 'ERROR', indent_level: int = 0):
        """
        Log an error message.

        Args:
            msg: The message to log.
            emoji_key: The key to retrieve the corresponding emoji.
            indent_level: The indentation level for the message.
        """
        self._display_func('ERROR', msg, emoji_key, indent_level)

    def critical(self, msg: str, emoji_key: Optional[str] = 'CRITICAL', indent_level: int = 0):
        """
        Log a critical message.

        Args:
            msg: The message to log.
            emoji_key: The key to retrieve the corresponding emoji.
            indent_level: The indentation level for the message.
        """
        self._display_func('CRITICAL', msg, emoji_key, indent_level)


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that uses tqdm.write to prevent progress bar disruption."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)