import sys
import os
import subprocess
import threading
import pandas as pd
import psutil
import GPUtil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton, 
                            QComboBox, QCheckBox, QSpinBox, QTextEdit, QProgressBar,
                            QFileDialog, QMessageBox, QSplitter, QFrame, QScrollArea,
                            QGridLayout, QRadioButton, QButtonGroup, QSlider, QDoubleSpinBox,
                            QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QFont, QPalette, QColor, QTextCursor, QIcon, QPainter, QLinearGradient
import json
import time

class ResourceMonitor(QThread):
    update_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        
    def run(self):
        while self.running:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_total_gb = memory.total / (1024**3)
                
                gpu_percent = 0
                gpu_memory_percent = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_percent = gpu.load * 100
                        gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                except:
                    pass
                
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                data = {
                    'cpu': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_used_gb': memory_used_gb,
                    'memory_total_gb': memory_total_gb,
                    'gpu': gpu_percent,
                    'gpu_memory': gpu_memory_percent,
                    'disk': disk_percent,
                    'timestamp': time.time()
                }
                
                self.update_signal.emit(data)
                
            except Exception as e:
                time.sleep(1)
    
    def stop(self):
        self.running = False

class InstallThread(QThread):
    output_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)
    package_signal = pyqtSignal(str)
    
    def __init__(self, packages, stop_event):
        super().__init__()
        self.packages = packages
        self.stop_event = stop_event
        
    def run(self):
        try:
            total_packages = len(self.packages)
            for i, package in enumerate(self.packages):
                if self.stop_event.is_set():
                    self.output_signal.emit("Installation stopped by user")
                    self.finished_signal.emit(False)
                    return
                    
                self.package_signal.emit(package)
                self.output_signal.emit(f"Installing {package}")
                
                progress = int((i / total_packages) * 100)
                self.progress_signal.emit(progress)
                
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "pip", "install", package],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    for line in process.stdout:
                        if self.stop_event.is_set():
                            process.terminate()
                            self.output_signal.emit("Installation stopped by user")
                            self.finished_signal.emit(False)
                            return
                        self.output_signal.emit(line)
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        self.output_signal.emit(f"{package} installed successfully")
                    else:
                        self.output_signal.emit(f"Failed to install {package}")
                        
                except Exception as e:
                    self.output_signal.emit(f"Error installing {package}: {str(e)}")
            
            self.progress_signal.emit(100)
            self.output_signal.emit("All dependencies installed successfully")
            self.finished_signal.emit(True)
            
        except Exception as e:
            self.output_signal.emit(f"Installation failed: {str(e)}")
            self.finished_signal.emit(False)

class WorkerThread(QThread):
    output_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
        self.interrupted = False
        self.process = None
        
    def run(self):
        try:
            self.output_signal.emit("Starting ML Pipeline")
            
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            
            for line in self.process.stdout:
                if self.interrupted:
                    break
                safe_line = line.encode('utf-8', errors='replace').decode('utf-8')
                self.output_signal.emit(safe_line)
                
            if not self.interrupted:
                self.process.wait()
                
            if self.interrupted:
                self.output_signal.emit("Pipeline interrupted by user")
                self.finished_signal.emit(False)
            elif self.process.returncode == 0:
                self.output_signal.emit("Pipeline completed successfully")
                self.finished_signal.emit(True)
            else:
                self.output_signal.emit(f"Pipeline failed with code {self.process.returncode}")
                self.finished_signal.emit(False)
                
        except Exception as e:
            safe_error = str(e).encode('utf-8', errors='replace').decode('utf-8')
            self.output_signal.emit(f"Error: {safe_error}")
            self.finished_signal.emit(False)
    
    def stop(self):
        self.interrupted = True
        if self.process:
            self.process.terminate()

class ResourceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(120)
        self.layout = QHBoxLayout(self)
        
        self.cpu_group = QGroupBox("CPU")
        cpu_layout = QVBoxLayout(self.cpu_group)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setFormat("%p%")
        self.cpu_label = QLabel("0%")
        cpu_layout.addWidget(self.cpu_bar)
        cpu_layout.addWidget(self.cpu_label)
        
        self.memory_group = QGroupBox("RAM")
        memory_layout = QVBoxLayout(self.memory_group)
        self.memory_bar = QProgressBar()
        self.memory_bar.setFormat("%p%")
        self.memory_label = QLabel("0 GB / 0 GB")
        memory_layout.addWidget(self.memory_bar)
        memory_layout.addWidget(self.memory_label)
        
        self.gpu_group = QGroupBox("GPU")
        gpu_layout = QVBoxLayout(self.gpu_group)
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setFormat("%p%")
        self.gpu_label = QLabel("Not available")
        gpu_layout.addWidget(self.gpu_bar)
        gpu_layout.addWidget(self.gpu_label)
        
        self.disk_group = QGroupBox("Disk")
        disk_layout = QVBoxLayout(self.disk_group)
        self.disk_bar = QProgressBar()
        self.disk_bar.setFormat("%p%")
        self.disk_label = QLabel("0%")
        disk_layout.addWidget(self.disk_bar)
        disk_layout.addWidget(self.disk_label)
        
        self.layout.addWidget(self.cpu_group)
        self.layout.addWidget(self.memory_group)
        self.layout.addWidget(self.gpu_group)
        self.layout.addWidget(self.disk_group)
    
    def update_resources(self, data):
        self.cpu_bar.setValue(int(data['cpu']))
        self.cpu_label.setText(f"{data['cpu']:.1f}%")
        
        self.memory_bar.setValue(int(data['memory_percent']))
        self.memory_label.setText(f"{data['memory_used_gb']:.1f}GB / {data['memory_total_gb']:.1f}GB")
        
        if data['gpu'] > 0:
            self.gpu_bar.setValue(int(data['gpu']))
            self.gpu_label.setText(f"{data['gpu']:.1f}%")
            self.gpu_group.setVisible(True)
        else:
            self.gpu_group.setVisible(False)
        
        self.disk_bar.setValue(int(data['disk']))
        self.disk_label.setText(f"{data['disk']:.1f}%")

class MLPipelineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Pipeline")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        if os.path.exists("icon.ico"):
            self.setWindowIcon(QIcon("icon.ico"))
        
        self.settings = QSettings("lastCFG/config.ini", QSettings.Format.IniFormat)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        self.resource_widget = ResourceWidget()
        main_layout.addWidget(self.resource_widget)
        
        self.create_tabs(main_layout)
        
        self.statusBar().showMessage("Ready")
        
        self.monitor = ResourceMonitor()
        self.monitor.update_signal.connect(self.resource_widget.update_resources)
        self.monitor.start()
        
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.current_progress = 0
        self.progress_stage = 0
        
        self.load_settings()
        
        self.apply_modern_light_theme()
        
    def closeEvent(self, event):
        self.save_settings()
        self.monitor.stop()
        self.monitor.wait(1000)
        event.accept()
    
    def save_settings(self):
        os.makedirs("lastCFG", exist_ok=True)
        
        self.settings.setValue("theme", self.theme_combo.currentText())
        self.settings.setValue("font_size", self.font_size_spin.value())
        self.settings.setValue("memory_limit", self.memory_limit_spin.value())
        self.settings.setValue("cpu_limit", self.cpu_limit_spin.value())
        self.settings.setValue("use_gpu", self.gpu_cb.isChecked())
        
        self.settings.setValue("last_zip_path", self.zip_path_edit.text())
        self.settings.setValue("last_train_path", self.train_path_edit.text())
        self.settings.setValue("last_test_path", self.test_path_edit.text())
        self.settings.setValue("last_output_dir", self.output_dir_edit.text())
        self.settings.setValue("last_model_path", self.model_path_edit.text())
        
        self.settings.setValue("last_target", self.target_edit.text())
        self.settings.setValue("last_algorithm", self.algorithm_combo.currentText())
        self.settings.setValue("last_task_type", self.task_type_combo.currentText())
        self.settings.setValue("last_output_columns", self.output_columns_edit.text())
        
        self.settings.sync()
    
    def load_settings(self):
        theme = self.settings.value("theme", "Modern Light")
        self.theme_combo.setCurrentText(theme)
        
        font_size = int(self.settings.value("font_size", 10))
        self.font_size_spin.setValue(font_size)
        
        memory_limit = float(self.settings.value("memory_limit", 8.0))
        self.memory_limit_spin.setValue(memory_limit)
        
        cpu_limit = int(self.settings.value("cpu_limit", 75))
        self.cpu_limit_spin.setValue(cpu_limit)
        
        use_gpu = self.settings.value("use_gpu", "false") == "true"
        self.gpu_cb.setChecked(use_gpu)
        
        self.zip_path_edit.setText(self.settings.value("last_zip_path", ""))
        self.train_path_edit.setText(self.settings.value("last_train_path", ""))
        self.test_path_edit.setText(self.settings.value("last_test_path", ""))
        self.output_dir_edit.setText(self.settings.value("last_output_dir", "results"))
        self.model_path_edit.setText(self.settings.value("last_model_path", ""))
        
        self.target_edit.setText(self.settings.value("last_target", ""))
        self.algorithm_combo.setCurrentText(self.settings.value("last_algorithm", "RandomForestClassifier"))
        self.task_type_combo.setCurrentText(self.settings.value("last_task_type", "Classification"))
        self.output_columns_edit.setText(self.settings.value("last_output_columns", ""))
    
    def create_tabs(self, main_layout):
        tabs = QTabWidget()
        
        quick_tab = self.create_quick_start_tab()
        tabs.addTab(quick_tab, "Quick Start")
        
        advanced_tab = self.create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced Settings")
        
        install_tab = self.create_install_tab()
        tabs.addTab(install_tab, "Dependencies")
        
        console_tab = self.create_console_tab()
        tabs.addTab(console_tab, "Console")
        
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "Settings")
        
        main_layout.addWidget(tabs)
    
    def create_quick_start_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        data_group = QGroupBox("Data Selection")
        data_layout = QGridLayout(data_group)
        
        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItems(["ZIP Archive", "Separate Train/Test Files"])
        self.data_mode_combo.currentTextChanged.connect(self.on_data_mode_changed)
        data_layout.addWidget(QLabel("Data Mode:"), 0, 0)
        data_layout.addWidget(self.data_mode_combo, 0, 1)
        
        self.zip_path_edit = QLineEdit()
        self.zip_browse_btn = QPushButton("Browse")
        self.zip_browse_btn.clicked.connect(self.browse_zip)
        data_layout.addWidget(QLabel("ZIP Archive:"), 1, 0)
        data_layout.addWidget(self.zip_path_edit, 1, 1)
        data_layout.addWidget(self.zip_browse_btn, 1, 2)
        
        self.train_path_edit = QLineEdit()
        self.train_browse_btn = QPushButton("Browse")
        self.train_browse_btn.clicked.connect(self.browse_train)
        
        self.test_path_edit = QLineEdit()
        self.test_browse_btn = QPushButton("Browse")
        self.test_browse_btn.clicked.connect(self.browse_test)
        
        data_layout.addWidget(QLabel("Train File:"), 2, 0)
        data_layout.addWidget(self.train_path_edit, 2, 1)
        data_layout.addWidget(self.train_browse_btn, 2, 2)
        
        data_layout.addWidget(QLabel("Test File:"), 3, 0)
        data_layout.addWidget(self.test_path_edit, 3, 1)
        data_layout.addWidget(self.test_browse_btn, 3, 2)
        
        layout.addWidget(data_group)
        
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout(model_group)
        
        self.target_edit = QLineEdit()
        model_layout.addWidget(QLabel("Target Column:"), 0, 0)
        model_layout.addWidget(self.target_edit, 0, 1)
        
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Classification", "Regression"])
        model_layout.addWidget(QLabel("Task Type:"), 1, 0)
        model_layout.addWidget(self.task_type_combo, 1, 1)
        
        self.algorithm_combo = QComboBox()
        self.update_algorithms()
        self.task_type_combo.currentTextChanged.connect(self.update_algorithms)
        model_layout.addWidget(QLabel("Algorithm:"), 2, 0)
        model_layout.addWidget(self.algorithm_combo, 2, 1)
        
        self.output_columns_edit = QLineEdit()
        self.output_columns_edit.setPlaceholderText("PassengerId, Survived")
        model_layout.addWidget(QLabel("Output Columns:"), 3, 0)
        model_layout.addWidget(self.output_columns_edit, 3, 1)
        
        layout.addWidget(model_group)
        
        mode_group = QGroupBox("Operation Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        self.mode_full = QRadioButton("Full Pipeline")
        self.mode_train_only = QRadioButton("Train Only")
        self.mode_test_only = QRadioButton("Test Only")
        
        self.mode_full.setChecked(True)
        
        mode_layout.addWidget(self.mode_full)
        mode_layout.addWidget(self.mode_train_only)
        mode_layout.addWidget(self.mode_test_only)
        
        layout.addWidget(mode_group)
        
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.clicked.connect(self.run_pipeline)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.stop_btn.setEnabled(False)
        
        self.open_results_btn = QPushButton("Open Results")
        self.open_results_btn.clicked.connect(self.open_results_dir)
        
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.open_results_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        progress_layout.addWidget(self.progress_bar, 4)
        progress_layout.addWidget(self.progress_label, 1)
        
        layout.addLayout(progress_layout)
        
        layout.addStretch()
        
        self.on_data_mode_changed("ZIP Archive")
        
        return widget

    def create_install_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        packages_group = QGroupBox("Required Packages")
        packages_layout = QVBoxLayout(packages_group)
        
        self.packages_table = QTableWidget()
        self.packages_table.setColumnCount(2)
        self.packages_table.setHorizontalHeaderLabels(["Package", "Version"])
        self.packages_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        packages = [
            ("pandas", ">=1.5.0"),
            ("numpy", ">=1.21.0"), 
            ("scikit-learn", ">=1.0.0"),
            ("tqdm", ">=4.60.0"),
            ("colorama", ">=0.4.0"),
            ("psutil", ">=5.8.0"),
            ("GPUtil", ">=1.4.0")
        ]
        
        self.packages_table.setRowCount(len(packages))
        for i, (pkg, ver) in enumerate(packages):
            self.packages_table.setItem(i, 0, QTableWidgetItem(pkg))
            self.packages_table.setItem(i, 1, QTableWidgetItem(ver))
        
        packages_layout.addWidget(self.packages_table)
        layout.addWidget(packages_group)
        
        install_group = QGroupBox("Installation")
        install_layout = QGridLayout(install_group)
        
        self.install_btn = QPushButton("Install All Dependencies")
        self.install_btn.clicked.connect(self.install_dependencies)
        
        self.stop_install_btn = QPushButton("Stop Installation")
        self.stop_install_btn.clicked.connect(self.stop_installation)
        self.stop_install_btn.setEnabled(False)
        
        self.current_package_label = QLabel("Ready to install")
        
        self.install_progress = QProgressBar()
        self.install_progress.setValue(0)
        
        install_layout.addWidget(self.install_btn, 0, 0)
        install_layout.addWidget(self.stop_install_btn, 0, 1)
        install_layout.addWidget(self.current_package_label, 1, 0, 1, 2)
        install_layout.addWidget(self.install_progress, 2, 0, 1, 2)
        
        layout.addWidget(install_group)
        
        install_console_group = QGroupBox("Installation Log")
        install_console_layout = QVBoxLayout(install_console_group)
        
        self.install_console = QTextEdit()
        self.install_console.setFont(QFont("Consolas", 9))
        self.install_console.setReadOnly(True)
        
        install_console_layout.addWidget(self.install_console)
        layout.addWidget(install_console_group)
        
        return widget

    def create_advanced_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QGridLayout(hyper_group)
        
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(1, 10)
        self.cv_folds_spin.setValue(3)
        hyper_layout.addWidget(QLabel("CV Folds:"), 0, 0)
        hyper_layout.addWidget(self.cv_folds_spin, 0, 1)
        
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 1000)
        self.random_state_spin.setValue(42)
        hyper_layout.addWidget(QLabel("Random State:"), 1, 0)
        hyper_layout.addWidget(self.random_state_spin, 1, 1)
        
        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(-1, 16)
        self.n_jobs_spin.setValue(-1)
        self.n_jobs_spin.setSpecialValueText("All cores")
        hyper_layout.addWidget(QLabel("Parallel Jobs:"), 2, 0)
        hyper_layout.addWidget(self.n_jobs_spin, 2, 1)
        
        left_layout.addWidget(hyper_group)
        
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        self.feature_engineering_cb = QCheckBox("Feature Engineering")
        self.feature_engineering_cb.setChecked(True)
        preprocess_layout.addWidget(self.feature_engineering_cb)
        
        self.handle_missing_cb = QCheckBox("Handle Missing Values")
        self.handle_missing_cb.setChecked(True)
        preprocess_layout.addWidget(self.handle_missing_cb)
        
        self.encode_categorical_cb = QCheckBox("Encode Categorical")
        self.encode_categorical_cb.setChecked(True)
        preprocess_layout.addWidget(self.encode_categorical_cb)
        
        self.tuning_cb = QCheckBox("Hyperparameter Tuning")
        self.tuning_cb.setChecked(True)
        preprocess_layout.addWidget(self.tuning_cb)
        
        left_layout.addWidget(preprocess_group)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout(output_group)
        
        self.output_dir_edit = QLineEdit("results")
        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        output_layout.addWidget(self.output_dir_edit, 0, 1)
        output_layout.addWidget(self.output_browse_btn, 0, 2)
        
        self.save_model_cb = QCheckBox("Save Trained Model")
        self.save_model_cb.setChecked(True)
        output_layout.addWidget(self.save_model_cb, 1, 0, 1, 3)
        
        self.save_weights_cb = QCheckBox("Save Model Weights")
        self.save_weights_cb.setChecked(False)
        output_layout.addWidget(self.save_weights_cb, 2, 0, 1, 3)
        
        self.no_view_cb = QCheckBox("Minimal Output")
        output_layout.addWidget(self.no_view_cb, 3, 0, 1, 3)
        
        right_layout.addWidget(output_group)
        
        test_only_group = QGroupBox("Test Only Options")
        test_only_layout = QGridLayout(test_only_group)
        
        self.model_path_edit = QLineEdit()
        self.model_browse_btn = QPushButton("Browse")
        self.model_browse_btn.clicked.connect(self.browse_model)
        test_only_layout.addWidget(QLabel("Model Path:"), 0, 0)
        test_only_layout.addWidget(self.model_path_edit, 0, 1)
        test_only_layout.addWidget(self.model_browse_btn, 0, 2)
        
        right_layout.addWidget(test_only_group)
        
        cache_group = QGroupBox("Cache Options")
        cache_layout = QHBoxLayout(cache_group)
        
        self.from_cache_cb = QCheckBox("Use Cached Data")
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        
        cache_layout.addWidget(self.from_cache_cb)
        cache_layout.addWidget(self.clear_cache_btn)
        
        right_layout.addWidget(cache_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400])
        
        layout.addWidget(splitter)
        
        return widget

    def create_console_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        console_controls = QHBoxLayout()
        
        self.clear_console_btn = QPushButton("Clear Console")
        self.clear_console_btn.clicked.connect(self.clear_console)
        
        self.export_log_btn = QPushButton("Export Log")
        self.export_log_btn.clicked.connect(self.export_log)
        
        console_controls.addWidget(self.clear_console_btn)
        console_controls.addWidget(self.export_log_btn)
        console_controls.addStretch()
        
        layout.addLayout(console_controls)
        
        self.console_output = QTextEdit()
        self.console_output.setFont(QFont("Consolas", 10))
        self.console_output.setReadOnly(True)
        
        layout.addWidget(self.console_output)
        
        return widget

    def create_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        theme_group = QGroupBox("Theme & Appearance")
        theme_layout = QGridLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Modern Light"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(QLabel("Theme:"), 0, 0)
        theme_layout.addWidget(self.theme_combo, 0, 1)
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setValue(10)
        self.font_size_spin.valueChanged.connect(self.change_font_size)
        theme_layout.addWidget(QLabel("Font Size:"), 1, 0)
        theme_layout.addWidget(self.font_size_spin, 1, 1)
        
        layout.addWidget(theme_group)
        
        resources_group = QGroupBox("System Resources")
        resources_layout = QGridLayout(resources_group)
        
        self.memory_limit_spin = QDoubleSpinBox()
        self.memory_limit_spin.setRange(0.1, 128.0)
        self.memory_limit_spin.setValue(8.0)
        self.memory_limit_spin.setSuffix(" GB")
        resources_layout.addWidget(QLabel("Memory Limit:"), 0, 0)
        resources_layout.addWidget(self.memory_limit_spin, 0, 1)
        
        self.cpu_limit_spin = QSpinBox()
        self.cpu_limit_spin.setRange(1, 100)
        self.cpu_limit_spin.setValue(75)
        self.cpu_limit_spin.setSuffix("%")
        resources_layout.addWidget(QLabel("CPU Limit:"), 1, 0)
        resources_layout.addWidget(self.cpu_limit_spin, 1, 1)
        
        self.gpu_cb = QCheckBox("Use GPU Acceleration")
        resources_layout.addWidget(self.gpu_cb, 2, 0, 1, 2)
        
        layout.addWidget(resources_group)
        
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        info_text.setHtml(f"""
        <h3>System Specifications</h3>
        <p><b>CPU Cores:</b> {cpu_count}</p>
        <p><b>Total RAM:</b> {memory_gb:.1f} GB</p>
        <p><b>Python:</b> {sys.version.split()[0]}</p>
        <p><b>Platform:</b> {sys.platform}</p>
        """)
        
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        return widget

    def update_algorithms(self):
        self.algorithm_combo.clear()
        
        if self.task_type_combo.currentText() == "Classification":
            algorithms = [
                "RandomForestClassifier",
                "LogisticRegression", 
                "SVC",
                "DecisionTreeClassifier",
                "KNeighborsClassifier",
                "GradientBoostingClassifier"
            ]
        else:
            algorithms = [
                "RandomForestRegressor",
                "LinearRegression",
                "SVR",
                "DecisionTreeRegressor", 
                "KNeighborsRegressor",
                "GradientBoostingRegressor"
            ]
        
        self.algorithm_combo.addItems(algorithms)

    def on_data_mode_changed(self, mode):
        is_zip_mode = mode == "ZIP Archive"
        
        self.zip_path_edit.setVisible(is_zip_mode)
        self.zip_browse_btn.setVisible(is_zip_mode)
        
        self.train_path_edit.setVisible(not is_zip_mode)
        self.train_browse_btn.setVisible(not is_zip_mode)
        self.test_path_edit.setVisible(not is_zip_mode)
        self.test_browse_btn.setVisible(not is_zip_mode)

    def browse_zip(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ZIP Archive", "", "ZIP Files (*.zip)"
        )
        if file_path:
            self.zip_path_edit.setText(file_path)

    def browse_train(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Train CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.train_path_edit.setText(file_path)

    def browse_test(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.test_path_edit.setText(file_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Pickle Files (*.pkl)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def open_results_dir(self):
        output_dir = self.output_dir_edit.text() or "results"
        if os.path.exists(output_dir):
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])
        else:
            QMessageBox.warning(self, "Warning", f"Directory {output_dir} does not exist")

    def install_dependencies(self):
        self.install_console.clear()
        self.log_to_install_console("Starting dependency installation")
        
        packages = []
        for i in range(self.packages_table.rowCount()):
            pkg = self.packages_table.item(i, 0).text()
            ver = self.packages_table.item(i, 1).text()
            packages.append(f"{pkg}{ver}")
        
        self.install_stop_event = threading.Event()
        
        self.install_thread = InstallThread(packages, self.install_stop_event)
        self.install_thread.output_signal.connect(self.log_to_install_console)
        self.install_thread.progress_signal.connect(self.install_progress.setValue)
        self.install_thread.package_signal.connect(self.current_package_label.setText)
        self.install_thread.finished_signal.connect(self.installation_finished)
        
        self.install_btn.setEnabled(False)
        self.stop_install_btn.setEnabled(True)
        self.install_thread.start()

    def stop_installation(self):
        if hasattr(self, 'install_thread') and self.install_thread.isRunning():
            self.install_stop_event.set()
            self.install_thread.wait(1000)
            self.log_to_install_console("Installation stopped by user")

    def installation_finished(self, success):
        self.install_btn.setEnabled(True)
        self.stop_install_btn.setEnabled(False)
        self.current_package_label.setText("Installation completed" if success else "Installation failed")

    def log_to_install_console(self, message):
        self.install_console.moveCursor(QTextCursor.MoveOperation.End)
        self.install_console.insertPlainText(message)
        if not message.endswith('\n'):
            self.install_console.insertPlainText('\n')
        self.install_console.moveCursor(QTextCursor.MoveOperation.End)

    def run_pipeline(self):
        command = [sys.executable, "main.py"]
        
        if self.output_dir_edit.text():
            command.extend(["--output_dir", self.output_dir_edit.text()])
        
        if self.cv_folds_spin.value() != 3:
            command.extend(["--cv_folds", str(self.cv_folds_spin.value())])
        
        if self.random_state_spin.value() != 42:
            command.extend(["--random_state", str(self.random_state_spin.value())])
        
        if self.n_jobs_spin.value() != -1:
            command.extend(["--n_jobs", str(self.n_jobs_spin.value())])
        
        if not self.feature_engineering_cb.isChecked():
            command.append("--no_feature_engineering")
        
        if not self.handle_missing_cb.isChecked():
            command.append("--no_handle_missing")
        
        if not self.encode_categorical_cb.isChecked():
            command.append("--no_encode_categorical")
        
        if not self.tuning_cb.isChecked():
            command.append("--no_tuning")
        
        if self.save_model_cb.isChecked():
            command.append("--save_model")
        
        if self.save_weights_cb.isChecked():
            command.append("--save_weights")
        
        if self.no_view_cb.isChecked():
            command.append("--no_view")
        
        if self.from_cache_cb.isChecked():
            command.append("--from_cache")
        
        if self.data_mode_combo.currentText() == "ZIP Archive" and self.zip_path_edit.text():
            command.extend(["--path", self.zip_path_edit.text()])
        else:
            if self.train_path_edit.text():
                command.extend(["--train", self.train_path_edit.text()])
            if self.test_path_edit.text():
                command.extend(["--test", self.test_path_edit.text()])
        
        if self.target_edit.text():
            command.extend(["--target", self.target_edit.text()])
        
        if self.task_type_combo.currentText() == "Classification":
            command.append("--classification")
        else:
            command.append("--regression")
        
        if self.algorithm_combo.currentText():
            command.extend(["--algorithm", self.algorithm_combo.currentText()])
        
        if self.output_columns_edit.text():
            columns = [col.strip() for col in self.output_columns_edit.text().split(",")]
            command.extend(["--output_columns"] + columns)
        
        if self.mode_train_only.isChecked():
            command.append("--only_train")
        elif self.mode_test_only.isChecked():
            command.append("--only_test")
            if self.model_path_edit.text():
                command.extend(["--model_path", self.model_path_edit.text()])
        
        self.log_to_console("Command: " + " ".join(command))
        
        self.worker_thread = WorkerThread(command)
        self.worker_thread.output_signal.connect(self.log_to_console)
        self.worker_thread.finished_signal.connect(self.pipeline_finished)
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting...")
        
        self.worker_thread.start()
        
        self.current_progress = 0
        self.progress_stage = 0
        self.progress_timer.start(500)

    def stop_pipeline(self):
        if hasattr(self, 'worker_thread'):
            self.worker_thread.stop()
            self.worker_thread.wait(3000)
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
            self.log_to_console("Pipeline stopped by user")
            self.pipeline_finished(False)

    def pipeline_finished(self, success):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_timer.stop()
        self.progress_label.setText("")
        
        if success:
            self.statusBar().showMessage("Pipeline completed successfully")
        else:
            self.statusBar().showMessage("Pipeline failed")

    def update_progress(self):
        stages = ["Extracting...", "Preprocessing...", "Training...", "Finalizing..."]
        self.current_progress = (self.current_progress + 2) % 100
        
        if self.current_progress >= 25 and self.progress_stage == 0:
            self.progress_stage = 1
        elif self.current_progress >= 50 and self.progress_stage == 1:
            self.progress_stage = 2
        elif self.current_progress >= 75 and self.progress_stage == 2:
            self.progress_stage = 3
        
        self.progress_bar.setValue(self.current_progress)
        self.progress_label.setText(f"{self.current_progress}% - {stages[self.progress_stage]}")

    def clear_cache(self):
        try:
            subprocess.run([sys.executable, "main.py", "--clear"], check=True)
            self.log_to_console("Cache cleared successfully")
        except subprocess.CalledProcessError as e:
            self.log_to_console(f"Failed to clear cache: {e}")

    def log_to_console(self, message):
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)
        self.console_output.insertPlainText(message)
        if not message.endswith('\n'):
            self.console_output.insertPlainText('\n')
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)

    def clear_console(self):
        self.console_output.clear()

    def export_log(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log File", "", "Text Files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.console_output.toPlainText())
                self.log_to_console(f"Log exported to {file_path}")
            except Exception as e:
                self.log_to_console(f"Failed to export log: {e}")

    def change_theme(self, theme_name):
        if theme_name == "Modern Light":
            self.apply_modern_light_theme()

    def apply_modern_light_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
                color: #212529;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: #ffffff;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                color: #495057;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #007bff;
                border-color: #dee2e6;
            }
            QTabBar::tab:hover {
                background-color: #dee2e6;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #ffffff;
                color: #212529;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #ffffff;
                color: #212529;
                border: 1px solid #ced4da;
                padding: 6px 8px;
                border-radius: 4px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border-color: #007bff;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #212529;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                background-color: #e9ecef;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                padding: 8px;
                border: 1px solid #dee2e6;
            }
        """)

    def change_font_size(self, size):
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ML Pipeline - GUI")
    app.setApplicationVersion("1.0.1")
    
    try:
        import PyQt6
        import pandas as pd
        import psutil
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        return 1
    
    window = MLPipelineGUI()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
