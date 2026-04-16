#!/usr/bin/env python3
"""Main entry point for Volleyball Action Annotator application."""
import os
import sys
from PyQt6.QtCore import QLibraryInfo
from PyQt6.QtWidgets import QApplication
from loguru import logger


def configure_qt_environment():
    """Keep Qt pointed at the PyQt6 runtime when cv2 is imported."""
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.path(
        QLibraryInfo.LibraryPath.PluginsPath
    )
    os.environ.pop("QT_QPA_FONTDIR", None)


configure_qt_environment()

from ui.main_window import VideoAnnotationApp

configure_qt_environment()


def setup_logging():
    """Configure logging."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "vaa.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )


def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger.info("Starting Volleyball Action Annotator")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Volleyball Action Annotator")
    app.setOrganizationName("VAA")
    
    # Set dark theme stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QPushButton {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #4c4c4c;
        }
        QPushButton:pressed {
            background-color: #1c1c1c;
        }
        QPushButton:disabled {
            background-color: #2b2b2b;
            color: #666666;
        }
        QLabel {
            color: #ffffff;
        }
        QTableWidget {
            background-color: #3c3c3c;
            alternate-background-color: #2b2b2b;
            selection-background-color: #007bff;
            gridline-color: #555555;
        }
        QHeaderView::section {
            background-color: #4c4c4c;
            color: #ffffff;
            padding: 4px;
            border: 1px solid #555555;
        }
        QComboBox {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 3px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
        }
        QCheckBox {
            color: #ffffff;
        }
        QMenuBar {
            background-color: #3c3c3c;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #007bff;
        }
        QMenu {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
        }
        QMenu::item:selected {
            background-color: #007bff;
        }
        QStatusBar {
            background-color: #3c3c3c;
            color: #ffffff;
        }
    """)
    
    # Create and show main window
    window = VideoAnnotationApp()
    window.show()
    
    # Run application event loop
    exit_code = app.exec()
    
    logger.info(f"Application exiting with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
