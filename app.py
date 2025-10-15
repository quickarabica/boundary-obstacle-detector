import sys
import math
import time
from typing import List, Optional, Tuple
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
                            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                            QCheckBox, QSlider, QRadioButton, QButtonGroup, QFrame, 
                            QSplitter, QGroupBox, QTextEdit)
from PyQt5.QtGui import (QPen, QBrush, QColor, QPainter, QPainterPath, QFont, 
                         QLinearGradient, QRadialGradient, QPolygonF, QTransform,
                         QPalette)
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, QPropertyAnimation, QEasingCurve

from geometry import convex_hull, hull_edges, intersection_point

Point = Tuple[float, float]


class CollisionGraphicsScene(QGraphicsScene):
    """Enhanced scene with better drawing capabilities"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QColor(25, 25, 35))
        
        # Grid properties
        self.grid_visible = True
        self.grid_size = 40
        self.grid_color = QColor(45, 45, 55)
        self.grid_major_color = QColor(60, 60, 70)
        
        # Visual settings
        self.hull_brush = QBrush(QColor(50, 100, 240, 50))
        self.hull_pen = QPen(QColor(65, 130, 255), 2)
        self.hull_pen.setCosmetic(True)
        
        self.path_pen = QPen(QColor(40, 200, 90), 3)
        self.path_pen.setCosmetic(True)
        
        self.obstacle_brush = QBrush(QColor(30, 41, 59))
        self.collision_pen = QPen(QColor(220, 38, 38), 2.5)
        self.collision_pen.setCosmetic(True)
        
    def setDarkMode(self, dark_mode: bool):
        """Update scene colors based on theme"""
        if dark_mode:
            self.setBackgroundBrush(QColor(25, 25, 35))
            self.grid_color = QColor(45, 45, 55)
            self.grid_major_color = QColor(60, 60, 70)
            self.hull_brush = QBrush(QColor(50, 100, 240, 50))
            self.hull_pen = QPen(QColor(65, 130, 255), 2)
            self.obstacle_brush = QBrush(QColor(30, 41, 59))
        else:
            self.setBackgroundBrush(QColor(240, 240, 245))
            self.grid_color = QColor(220, 220, 220)
            self.grid_major_color = QColor(200, 200, 200)
            self.hull_brush = QBrush(QColor(100, 150, 255, 50))
            self.hull_pen = QPen(QColor(65, 130, 255), 2)
            self.obstacle_brush = QBrush(QColor(70, 80, 100))
            
        self.hull_pen.setCosmetic(True)
        
    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw a professional grid background with subtle gradients"""
        super().drawBackground(painter, rect)
        
        if not self.grid_visible:
            return
            
        # Draw terrain-like background with subtle gradient
        gradient = QLinearGradient(0, 0, 0, rect.height())
        background_color = self.backgroundBrush().color()
        slightly_darker = QColor(
            max(0, background_color.red() - 5),
            max(0, background_color.green() - 5),
            max(0, background_color.blue() - 5)
        )
        gradient.setColorAt(0, background_color)
        gradient.setColorAt(1, slightly_darker)
        painter.fillRect(rect, gradient)
        
        # Determine grid range
        left = int(rect.left()) - (int(rect.left()) % self.grid_size)
        top = int(rect.top()) - (int(rect.top()) % self.grid_size)
        
        # Draw minor grid lines
        painter.setPen(QPen(self.grid_color, 1))
        for x in range(left, int(rect.right()), self.grid_size):
            painter.drawLine(x, rect.top(), x, rect.bottom())
        for y in range(top, int(rect.bottom()), self.grid_size):
            painter.drawLine(rect.left(), y, rect.right(), y)
            
        # Draw major grid lines
        painter.setPen(QPen(self.grid_major_color, 1))
        for x in range(left, int(rect.right()), self.grid_size * 5):
            painter.drawLine(x, rect.top(), x, rect.bottom())
        for y in range(top, int(rect.bottom()), self.grid_size * 5):
            painter.drawLine(rect.left(), y, rect.right(), y)


class CollisionDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collision Explorer")
        self.resize(1200, 700)
        
        # Data structures
        self.points: List[Point] = []
        self.hull: List[Point] = []
        self.path_start: Optional[Point] = None
        self.path_end: Optional[Point] = None
        self.collisions: List[Point] = []
        
        # Simulation state
        self.car_t: float = 0.0  # param along path [0,1]
        self.anim_running: bool = False
        self._last_ts: Optional[float] = None
        self.first_collision_t: Optional[float] = None
        self.mode = "obstacle"  # "obstacle", "start", "end"
        
        # Car path trail
        self.trail_points: List[Point] = []
        self.trail_max_points = 30
        self.show_motion_trail = True
        
        # Theme state
        self.dark_mode = True
        
        # Initialize UI
        self._init_ui()
        self._connect_signals()
        
        # Apply initial theme
        self._apply_theme()
        
        # Start the UI refresh timer
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh_info)
        self._refresh_timer.start(500)  # Update info every 500ms when not animating
        
    def _init_ui(self):
        """Initialize the modern UI layout"""
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left side: Graphics view
        self.view_container = QWidget()
        view_layout = QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scene = CollisionGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setBackgroundBrush(QBrush(QColor(25, 25, 35)))
        
        view_layout.addWidget(self.view)
        self.splitter.addWidget(self.view_container)
        
        # Right side: Controls panel
        self.panel = QWidget()
        self.panel.setMinimumWidth(300)
        self.panel.setMaximumWidth(450)
        panel_layout = QVBoxLayout(self.panel)
        
        # Title
        title_label = QLabel("Collision Explorer")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title_label)
        
        # Mode selection group
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_buttons = QButtonGroup(self)
        
        self.obstacle_radio = QRadioButton("Add obstacle")
        self.obstacle_radio.setChecked(True)
        self.start_radio = QRadioButton("Set path start")
        self.end_radio = QRadioButton("Set path end")
        
        self.mode_buttons.addButton(self.obstacle_radio)
        self.mode_buttons.addButton(self.start_radio)
        self.mode_buttons.addButton(self.end_radio)
        
        mode_layout.addWidget(self.obstacle_radio)
        mode_layout.addWidget(self.start_radio)
        mode_layout.addWidget(self.end_radio)
        
        panel_layout.addWidget(mode_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        self.clear_path_btn = QPushButton("Clear Path")
        self.clear_all_btn = QPushButton("Clear All")
        
        buttons_layout.addWidget(self.clear_path_btn)
        buttons_layout.addWidget(self.clear_all_btn)
        panel_layout.addLayout(buttons_layout)
        
        # Simulation controls
        sim_group = QGroupBox("Simulation")
        sim_layout = QVBoxLayout(sim_group)
        
        # Play controls
        play_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.reset_btn = QPushButton("↺ Reset")
        
        play_layout.addWidget(self.play_btn)
        play_layout.addWidget(self.pause_btn)
        play_layout.addWidget(self.reset_btn)
        sim_layout.addLayout(play_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(20)
        self.speed_slider.setMaximum(600)
        self.speed_slider.setValue(160)
        speed_layout.addWidget(self.speed_slider)
        self.speed_value = QLabel("160 px/s")
        speed_layout.addWidget(self.speed_value)
        sim_layout.addLayout(speed_layout)
        
        # Hit tolerance
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Hit tolerance:"))
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(50)
        self.tolerance_slider.setValue(15)
        self.tolerance_slider.setSingleStep(1)
        self.hit_tol = 1.5  # Initial value
        tolerance_layout.addWidget(self.tolerance_slider)
        self.tolerance_value = QLabel("1.5 px")
        tolerance_layout.addWidget(self.tolerance_value)
        sim_layout.addLayout(tolerance_layout)
        
        # Options
        self.stop_on_collision = QCheckBox("Stop at first collision")
        self.stop_on_collision.setChecked(True)
        sim_layout.addWidget(self.stop_on_collision)
        
        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(True)
        sim_layout.addWidget(self.show_grid)
        
        self.show_trail = QCheckBox("Show motion trail")
        self.show_trail.setChecked(True)
        sim_layout.addWidget(self.show_trail)
        
        # Hotkey info
        hotkeys_label = QLabel("Hotkeys: Space=Play/Pause, R=Reset")
        hotkeys_label.setStyleSheet("color: gray;")
        sim_layout.addWidget(hotkeys_label)
        
        panel_layout.addWidget(sim_group)
        
        # Info panel
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(200)
        info_layout.addWidget(self.info_text)
        
        panel_layout.addWidget(info_group)
        
        # Status indicators
        self.status_layout = QHBoxLayout()
        self.collision_status = QLabel("No collision")
        self.collision_status.setStyleSheet("color: green; font-weight: bold;")
        self.status_layout.addWidget(self.collision_status)
        
        self.path_status = QLabel("No path")
        self.status_layout.addWidget(self.path_status)
        panel_layout.addLayout(self.status_layout)
        
        # Add theme toggle at the bottom of the panel
        theme_layout = QHBoxLayout()
        self.dark_mode_checkbox = QCheckBox("Dark mode")
        self.dark_mode_checkbox.setChecked(True)
        theme_layout.addWidget(self.dark_mode_checkbox)
        panel_layout.addLayout(theme_layout)
        
        # Add the panel to the splitter
        self.splitter.addWidget(self.panel)
        
        # Set splitter initial sizes
        self.splitter.setSizes([800, 400])
        
        # Init the scene
        self.scene.setSceneRect(0, 0, 800, 600)
        self._redraw()
        
    def _connect_signals(self):
        """Connect UI signals to slots"""
        # Mode selection
        self.obstacle_radio.toggled.connect(lambda: self._set_mode("obstacle"))
        self.start_radio.toggled.connect(lambda: self._set_mode("start"))
        self.end_radio.toggled.connect(lambda: self._set_mode("end"))
        
        # Button actions
        self.clear_path_btn.clicked.connect(self._clear_path)
        self.clear_all_btn.clicked.connect(self._clear_all)
        
        # Simulation controls
        self.play_btn.clicked.connect(self._start_animation)
        self.pause_btn.clicked.connect(self._pause_animation)
        self.reset_btn.clicked.connect(self._reset_animation)
        
        # Sliders
        self.speed_slider.valueChanged.connect(self._update_speed)
        self.tolerance_slider.valueChanged.connect(self._update_tolerance)
        
        # Options
        self.show_grid.stateChanged.connect(self._toggle_grid)
        self.show_trail.stateChanged.connect(self._toggle_trail)
        
        # Mouse click in view
        self.view.mousePressEvent = self._handle_view_click
        
        # Keyboard shortcuts
        self.shortcut_play = Qt.Key_Space
        self.shortcut_reset = Qt.Key_R
        
        # Theme toggle
        self.dark_mode_checkbox.stateChanged.connect(self._toggle_theme)
        
    def _set_mode(self, mode):
        """Set the interaction mode"""
        self.mode = mode
        
    def _update_speed(self, value):
        """Update animation speed"""
        self.speed_value.setText(f"{value} px/s")
        
    def _update_tolerance(self, value):
        """Update hit tolerance"""
        self.hit_tol = value / 10.0
        self.tolerance_value.setText(f"{self.hit_tol} px")
        self._update_collisions()
        self._redraw()
        
    def _toggle_grid(self, state):
        """Toggle grid visibility"""
        self.scene.grid_visible = (state == Qt.Checked)
        self.view.viewport().update()
        
    def _toggle_trail(self, state):
        """Toggle motion trail visibility"""
        self.show_motion_trail = (state == Qt.Checked)
        self._redraw()
        
    def _toggle_theme(self, state):
        """Toggle between light and dark themes"""
        self.dark_mode = (state == Qt.Checked)
        self._apply_theme()
        
    def _apply_theme(self):
        """Apply the current theme to all UI elements"""
        app = QApplication.instance()
        
        if self.dark_mode:
            # Dark theme palette
            palette = app.palette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
            
            # Theme-specific styles
            self.collision_status.setStyleSheet(
                "color: green; font-weight: bold;" if not self.collisions else "color: red; font-weight: bold;"
            )
            self.view.setBackgroundBrush(QBrush(QColor(25, 25, 35)))
        else:
            # Light theme palette
            palette = app.palette()
            palette.setColor(QPalette.Window, QColor(240, 240, 245))
            palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.AlternateBase, QColor(233, 233, 233))
            palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
            palette.setColor(QPalette.Text, QColor(0, 0, 0))
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(0, 102, 204))
            palette.setColor(QPalette.Highlight, QColor(61, 174, 233))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            
            # Theme-specific styles
            self.collision_status.setStyleSheet(
                "color: green; font-weight: bold;" if not self.collisions else "color: red; font-weight: bold;"
            )
            self.view.setBackgroundBrush(QBrush(QColor(240, 240, 245)))
            
        app.setPalette(palette)
        
        # Update scene colors
        self.scene.setDarkMode(self.dark_mode)
        self.view.viewport().update()
        
        # Redraw with new theme colors
        self._redraw()
    
    def _handle_view_click(self, event):
        """Handle mouse clicks in the view"""
        # Convert view coordinates to scene coordinates
        scene_pos = self.view.mapToScene(event.pos())
        pt = (scene_pos.x(), scene_pos.y())
        
        if self.mode == "obstacle":
            self.points.append(pt)
            self.hull = convex_hull(self.points)
        elif self.mode == "start":
            self.path_start = pt
            self._reset_animation()
        elif self.mode == "end":
            self.path_end = pt
            self._reset_animation()
            
        self._update_collisions()
        self._redraw()
        self._refresh_info()
        
        # Call the parent class's mousePressEvent
        super(QGraphicsView, self.view).mousePressEvent(event)
        
    def _update_collisions(self):
        """Detect collisions between path and hull"""
        self.collisions.clear()
        self.first_collision_t = None
        
        if not self.path_start or not self.path_end:
            return
            
        path_segment = (self.path_start, self.path_end)
        
        # Check for intersections with hull edges
        for edge in hull_edges(self.hull):
            pt = intersection_point(path_segment, edge)
            if pt:
                if not any((abs(pt[0] - q[0]) < 1e-6 and abs(pt[1] - q[1]) < 1e-6) for q in self.collisions):
                    self.collisions.append(pt)
                    
        # Check endpoints with tolerance
        tol = self.hit_tol
        for edge in hull_edges(self.hull):
            for endpoint in (self.path_start, self.path_end):
                if self._point_on_segment(edge[0], edge[1], endpoint, tol):
                    if not any((abs(endpoint[0] - q[0]) < 1e-6 and abs(endpoint[1] - q[1]) < 1e-6) for q in self.collisions):
                        self.collisions.append(endpoint)
                        
        # Find first collision
        self.first_collision_t = self._compute_first_collision_t(self.collisions)
        
        # Update collision status indicator
        if self.collisions:
            self.collision_status.setText(f"Collision detected ({len(self.collisions)})")
            self.collision_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.collision_status.setText("No collision")
            self.collision_status.setStyleSheet("color: green; font-weight: bold;")
            
    def _redraw(self):
        """Redraw the entire scene"""
        self.scene.clear()
        
        # Draw hull
        if len(self.hull) >= 3:
            hull_polygon = QPolygonF()
            for x, y in self.hull:
                hull_polygon.append(QPointF(x, y))
            hull_item = self.scene.addPolygon(hull_polygon, self.scene.hull_pen, self.scene.hull_brush)
            hull_item.setZValue(10)
        elif len(self.hull) == 2:
            p1, p2 = self.hull
            self.scene.addLine(p1[0], p1[1], p2[0], p2[1], self.scene.hull_pen)
            
        # Draw obstacle points
        for x, y in self.points:
            point_item = self.scene.addEllipse(x-4, y-4, 8, 8, QPen(Qt.NoPen), self.scene.obstacle_brush)
            point_item.setZValue(20)
            
        # Draw path
        if self.path_start and self.path_end:
            # Draw path line
            path_item = self.scene.addLine(
                self.path_start[0], self.path_start[1],
                self.path_end[0], self.path_end[1],
                self.scene.path_pen
            )
            path_item.setZValue(30)
            
            # Draw endpoints
            start_item = self.scene.addEllipse(
                self.path_start[0]-5, self.path_start[1]-5, 10, 10, 
                QPen(QColor(40, 200, 90), 1.5), QBrush(Qt.NoBrush)
            )
            start_item.setZValue(31)
            
            end_item = self.scene.addEllipse(
                self.path_end[0]-5, self.path_end[1]-5, 10, 10, 
                QPen(QColor(40, 200, 90), 1.5), QBrush(Qt.NoBrush)
            )
            end_item.setZValue(31)
            
            # Draw motion trail if enabled
            if self.show_motion_trail and self.trail_points:
                for i, (x, y) in enumerate(self.trail_points):
                    alpha = int(160 * (i / len(self.trail_points)))
                    trail_brush = QBrush(QColor(40, 180, 100, alpha))
                    trail_size = 3 + 4 * (i / len(self.trail_points))
                    trail_item = self.scene.addEllipse(
                        x - trail_size/2, y - trail_size/2, 
                        trail_size, trail_size, 
                        QPen(Qt.NoPen), trail_brush
                    )
                    trail_item.setZValue(35)
                    
            # Draw car
            self._draw_car()
            
        # Draw collisions
        for x, y in self.collisions:
            collision_item = self.scene.addEllipse(x-7, y-7, 14, 14, self.scene.collision_pen, QBrush(Qt.NoBrush))
            collision_item.setZValue(40)
            
        # Highlight first collision
        if self.first_collision_t is not None and self.path_start and self.path_end:
            cx, cy = self._point_at_t(self.first_collision_t)
            first_collision = self.scene.addEllipse(
                cx-8, cy-8, 16, 16, 
                QPen(QColor(185, 28, 28), 3), QBrush(Qt.NoBrush)
            )
            first_collision.setZValue(50)
            
            # Add a pulsing effect
            if self.anim_running and self.car_t >= self.first_collision_t:
                glow = self.scene.addEllipse(
                    cx-15, cy-15, 30, 30,
                    QPen(QColor(220, 38, 38, 100), 2), 
                    QBrush(QColor(220, 38, 38, 30))
                )
                glow.setZValue(45)
            
    def _draw_car(self):
        """Draw a more detailed car at the current position"""
        if not self.path_start or not self.path_end:
            return
            
        pos = self._point_at_t(self.car_t)
        dx, dy = self._path_vector()
        if dx == 0 and dy == 0:
            return
            
        # Calculate car orientation
        ang = math.atan2(dy, dx)
        
        # Check if car is in collision state
        in_collision = (self.first_collision_t is not None and 
                       self.car_t >= self.first_collision_t - 1e-6)
        
        # Car body with wheels
        car_group = self.scene.createItemGroup([])
        car_group.setZValue(60)
        
        # Add to trail
        self.trail_points.append(pos)
        if len(self.trail_points) > self.trail_max_points:
            self.trail_points.pop(0)
        
        # Create car body
        length, width = 18.0, 10.0
        
        # Car color changes on collision
        body_color = QColor(200, 40, 40) if in_collision else QColor(15, 118, 110)
        wheel_color = QColor(30, 30, 30)
        
        # Create a car shape as a path
        car_path = QPainterPath()
        
        # Car body points (designed in local space)
        nose_x, nose_y = length/2, 0
        rear_x, rear_y = -length/2, 0
        left_x, left_y = 0, width/2
        right_x, right_y = 0, -width/2
        
        car_path.moveTo(nose_x, 0)  # nose
        car_path.quadTo(nose_x-length/4, -width/2, 0, -width/2)  # right curve
        car_path.lineTo(-length/3, -width/2)  # right side
        car_path.quadTo(-length/2, -width/2, -length/2, 0)  # back right
        car_path.quadTo(-length/2, width/2, -length/3, width/2)  # back left
        car_path.lineTo(0, width/2)  # left side
        car_path.quadTo(nose_x-length/4, width/2, nose_x, 0)  # left curve to nose
        
        # Transform car to world space
        transform = QTransform()
        transform.translate(pos[0], pos[1])
        transform.rotate(math.degrees(ang))
        
        car_path = transform.map(car_path)
        
        # Add car body
        car_item = self.scene.addPath(car_path, QPen(QColor(0, 0, 0)), QBrush(body_color))
        car_group.addToGroup(car_item)
        
        # Add highlight/shadow for 3D effect
        if not in_collision:
            highlight_path = QPainterPath()
            highlight_path.moveTo(nose_x, 0)
            highlight_path.quadTo(nose_x-length/4, -width/4, 0, -width/3)
            highlight_path.lineTo(0, -width/5)
            highlight_path.quadTo(nose_x-length/4, -width/6, nose_x-width/3, 0)
            highlight_path = transform.map(highlight_path)
            
            highlight = self.scene.addPath(highlight_path, QPen(Qt.NoPen), QBrush(QColor(30, 150, 140)))
            car_group.addToGroup(highlight)
            
        # Add wheels - just small black circles
        wheel_size = 4.0
        wheel_offset_x = length/3
        wheel_offset_y = width/2 + wheel_size/4
        
        # Transform wheel positions
        fl_wheel_pos = transform.map(QPointF(wheel_offset_x, -wheel_offset_y))
        fr_wheel_pos = transform.map(QPointF(wheel_offset_x, wheel_offset_y))
        rl_wheel_pos = transform.map(QPointF(-wheel_offset_x, -wheel_offset_y))
        rr_wheel_pos = transform.map(QPointF(-wheel_offset_x, wheel_offset_y))
        
        # Add wheels
        for wheel_pos in [fl_wheel_pos, fr_wheel_pos, rl_wheel_pos, rr_wheel_pos]:
            wheel = self.scene.addEllipse(
                wheel_pos.x()-wheel_size/2, wheel_pos.y()-wheel_size/2,
                wheel_size, wheel_size,
                QPen(Qt.NoPen), QBrush(wheel_color)
            )
            car_group.addToGroup(wheel)
            
        # Add a windshield
        windshield_path = QPainterPath()
        windshield_path.moveTo(length/6, 0)
        windshield_path.lineTo(0, -width/3)
        windshield_path.lineTo(0, width/3)
        windshield_path.lineTo(length/6, 0)
        windshield_path = transform.map(windshield_path)
        
        windshield = self.scene.addPath(
            windshield_path, QPen(Qt.NoPen), 
            QBrush(QColor(150, 230, 255) if not in_collision else QColor(255, 200, 200))
        )
        car_group.addToGroup(windshield)
            
        # If collision occurred, add effect
        if in_collision:
            explosion_radius = 20.0
            explosion = self.scene.addEllipse(
                pos[0]-explosion_radius, pos[1]-explosion_radius,
                explosion_radius*2, explosion_radius*2,
                QPen(QColor(255, 100, 30, 150), 2),
                QBrush(QColor(255, 140, 50, 100))
            )
            car_group.addToGroup(explosion)
            
    def _clear_path(self):
        """Clear the path"""
        self.path_start = None
        self.path_end = None
        self.collisions.clear()
        self.trail_points.clear()
        
        # Stop animation
        self._pause_animation()
        self.car_t = 0.0
        self.first_collision_t = None
        
        self._redraw()
        self._refresh_info()
        
        # Update path status
        self.path_status.setText("No path")
        
    def _clear_all(self):
        """Clear everything"""
        self.points.clear()
        self.hull.clear()
        self._clear_path()
        
    def _start_animation(self):
        """Start the car animation"""
        if not (self.path_start and self.path_end):
            return
            
        if self.car_t >= 1.0:
            self.car_t = 0.0
            self.trail_points.clear()
            
        self.anim_running = True
        self._last_ts = time.perf_counter()
        self._tick()
        
    def _pause_animation(self):
        """Pause the car animation"""
        self.anim_running = False
        self._last_ts = None
        
    def _reset_animation(self):
        """Reset car to start position"""
        self._pause_animation()
        self.car_t = 0.0
        self.trail_points.clear()
        self._redraw()
        self._refresh_info()
        
    def _tick(self):
        """Animation tick handler"""
        if not self.anim_running or not (self.path_start and self.path_end):
            return
            
        now = time.perf_counter()
        dt = (now - self._last_ts) if self._last_ts else 0.0
        self._last_ts = now
        
        L = self._path_length()
        if L <= 1e-9:
            self._pause_animation()
            return
            
        # Advance car position
        step_t = (self.speed_slider.value() * dt) / L
        target_t = self.car_t + step_t
        
        # Check for collision
        if self.stop_on_collision.isChecked() and self.first_collision_t is not None:
            if target_t >= self.first_collision_t:
                self.car_t = self.first_collision_t
                self._redraw()
                self._refresh_info()
                self._pause_animation()
                return
                
        # Update position
        self.car_t = min(target_t, 1.0)
        if self.car_t >= 1.0:
            self._redraw()
            self._refresh_info()
            self._pause_animation()
            return
            
        self._redraw()
        # Don't refresh info every frame for performance
        if dt > 0.1:
            self._refresh_info()
            
        # Continue animation
        QTimer.singleShot(16, self._tick)  # ~60 FPS
        
    def _refresh_info(self):
        """Update info panel with current state"""
        lines = [
            f"<b>Obstacles:</b> {len(self.points)}",
            f"<b>Hull vertices:</b> {len(self.hull)}",
            f"<b>Path start:</b> {self._fmt(self.path_start)}",
            f"<b>Path end:</b> {self._fmt(self.path_end)}",
            f"<b>Start position:</b> {self._pos_str(self.path_start)}",
            f"<b>End position:</b> {self._pos_str(self.path_end)}",
            f"<b>Car t:</b> {self.car_t:.3f}" if self.path_start and self.path_end else "<b>Car t:</b> —",
            "",
            "<b>Collisions:</b>"
        ]
        
        if self.collisions:
            lines.extend(f"• {self._fmt(p)}" for p in sorted(self.collisions, key=lambda p: (p[0], p[1])))
            if self.first_collision_t is not None:
                lines.append(f"<b>First collision at t={self.first_collision_t:.3f}</b>")
        else:
            lines.append("• None")
            
        self.info_text.setHtml("<p>" + "<br>".join(lines) + "</p>")
        
        # Update path status
        if self.path_start and self.path_end:
            self.path_status.setText(f"Path: {self._fmt_short(self.path_start)} → {self._fmt_short(self.path_end)}")
        else:
            self.path_status.setText("No path")
            
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == self.shortcut_play:  # Space
            if self.anim_running:
                self._pause_animation()
            else:
                self._start_animation()
        elif event.key() == self.shortcut_reset:  # R
            self._reset_animation()
        else:
            super().keyPressEvent(event)
        
    # Geometric helpers
    def _path_vector(self) -> Tuple[float, float]:
        """Get the path vector"""
        if not (self.path_start and self.path_end):
            return (0.0, 0.0)
        return (self.path_end[0] - self.path_start[0], self.path_end[1] - self.path_start[1])
        
    def _path_length(self) -> float:
        """Calculate path length"""
        dx, dy = self._path_vector()
        return math.hypot(dx, dy)
        
    def _point_at_t(self, t: float) -> Point:
        """Get point along path at parameter t"""
        if not (self.path_start and self.path_end):
            return (0.0, 0.0)
        x0, y0 = self.path_start
        dx, dy = self._path_vector()
        return (x0 + dx * t, y0 + dy * t)
        
    def _compute_first_collision_t(self, hits: List[Point]) -> Optional[float]:
        """Find the earliest collision along the path"""
        if not (self.path_start and self.path_end) or not hits:
            return None
            
        x0, y0 = self.path_start
        dx, dy = self._path_vector()
        L2 = dx * dx + dy * dy
        
        if L2 <= 1e-12:
            return None
            
        best: Optional[float] = None
        for hx, hy in hits:
            t = ((hx - x0) * dx + (hy - y0) * dy) / L2
            if 0.0 <= t <= 1.0:
                best = t if best is None or t < best else best
                
        return best
        
    def _point_in_convex_hull(self, p: Optional[Point], eps: float = 1e-9) -> Optional[str]:
        """Check if point is inside/on/outside the hull"""
        if p is None or len(self.hull) == 0:
            return None
            
        if len(self.hull) == 1:
            return "on boundary" if math.hypot(p[0] - self.hull[0][0], p[1] - self.hull[0][1]) <= eps else "outside"
            
        if len(self.hull) == 2:
            return "on boundary" if self._point_on_segment(self.hull[0], self.hull[1], p, eps) else "outside"
            
        on_boundary = False
        n = len(self.hull)
        
        for i in range(n):
            a = self.hull[i]
            b = self.hull[(i + 1) % n]
            
            # Check if on boundary
            if self._point_on_segment(a, b, p, max(eps, self.hit_tol)):
                on_boundary = True
                continue
                
            # Check if inside (CCW hull => cross >= -eps)
            if self._cross(b[0] - a[0], b[1] - a[1], p[0] - a[0], p[1] - a[1]) < -eps:
                return "outside"
                
        return "on boundary" if on_boundary else "inside"
    
    @staticmethod
    def _cross(ax: float, ay: float, bx: float, by: float) -> float:
        """2D cross product"""
        return ax * by - ay * bx
    
    def _point_on_segment(self, a: Point, b: Point, p: Point, eps: float) -> bool:
        """Check if point is on segment with tolerance"""
        (ax, ay), (bx, by), (px, py) = a, b, p
        abx, aby = (bx - ax), (by - ay)
        apx, apy = (px - ax), (py - ay)
        
        ab2 = abx * abx + aby * aby
        if ab2 == 0.0:
            return math.hypot(px - ax, py - ay) <= eps
            
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        qx, qy = ax + t * abx, ay + t * aby
        
        return math.hypot(px - qx, py - qy) <= eps
    
    def _pos_str(self, p: Optional[Point]) -> str:
        """Get position description"""
        state = self._point_in_convex_hull(p)
        return state if state is not None else "—"
    
    @staticmethod
    def _fmt(pt: Optional[Point]) -> str:
        """Format point coordinates"""
        return f"({pt[0]:.1f}, {pt[1]:.1f})" if pt else "—"
        
    @staticmethod
    def _fmt_short(pt: Optional[Point]) -> str:
        """Format point coordinates (shorter)"""
        return f"({int(pt[0])},{int(pt[1])})" if pt else "—"


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show window (theme will be set in constructor)
    window = CollisionDetectorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
