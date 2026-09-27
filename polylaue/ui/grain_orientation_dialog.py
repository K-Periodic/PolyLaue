# Copyright © 2026, UChicago Argonne, LLC. See "LICENSE" for full details.

"""Interactive grain-orientation and cubic CSL report."""

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from polylaue.model.core import (
    OrientationRelationship,
    analyze_orientation_relationship,
    apply_angular_shift,
    match_cubic_csls,
    rotation_matrix_residuals,
)
from polylaue.model.reflections.external import ExternalReflections
from polylaue.ui.reflections_editor import ReflectionsEditor
from polylaue.ui.utils.keep_dialog_on_top import keep_dialog_on_top


@dataclass(frozen=True)
class _SelectedMatrix:
    abc: np.ndarray
    source: str
    warning: str | None = None


class GrainOrientationDialog(QDialog):
    """Compare two indexed grains and check cubic CSL relationships."""

    def __init__(
        self,
        reflections_editor: ReflectionsEditor,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        keep_dialog_on_top(self)
        self.reflections_editor = reflections_editor

        self.setWindowTitle('Find Grain Orientation and Cubic CSL')
        self.setMinimumSize(880, 620)
        self.resize(1000, 720)

        self._build_ui()
        self._connect_signals()
        self._populate_grains()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        explanation = QLabel(
            'Select two indexed grains. The calculation reports the raw '
            'rotation, the minimum cubic disorientation, its crystallographic '
            'axis in both grains, and full-rotation matches to exact cubic CSLs.'
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        controls_group = QGroupBox('Grains and CSL settings')
        controls = QGridLayout(controls_group)

        self.grain_1 = QComboBox()
        self.grain_2 = QComboBox()
        self.use_current_scan = QCheckBox(
            'Use tracked orientation at the displayed scan when available'
        )
        self.use_current_scan.setChecked(True)

        self.max_sigma = QSpinBox()
        self.max_sigma.setRange(1, 99)
        self.max_sigma.setValue(49)
        self.max_sigma.setToolTip(
            'Generate all symmetry-distinct exact cubic CSL rotations up to '
            'this coincidence index.'
        )

        self.brandon_constant = QDoubleSpinBox()
        self.brandon_constant.setRange(0.1, 90.0)
        self.brandon_constant.setDecimals(2)
        self.brandon_constant.setValue(15.0)
        self.brandon_constant.setSuffix('°')
        self.brandon_constant.setToolTip(
            'The empirical Brandon limit is this value divided by sqrt(Sigma).'
        )

        self.find_button = QPushButton('Find Grain Orientation')
        self.find_button.setDefault(True)

        controls.addWidget(QLabel('Grain 1:'), 0, 0)
        controls.addWidget(self.grain_1, 0, 1)
        controls.addWidget(QLabel('Grain 2:'), 0, 2)
        controls.addWidget(self.grain_2, 0, 3)
        controls.addWidget(self.use_current_scan, 1, 0, 1, 4)
        controls.addWidget(QLabel('Maximum Σ:'), 2, 0)
        controls.addWidget(self.max_sigma, 2, 1)
        controls.addWidget(QLabel('Brandon constant:'), 2, 2)
        controls.addWidget(self.brandon_constant, 2, 3)
        controls.addWidget(self.find_button, 3, 0, 1, 4)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        layout.addWidget(controls_group)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.orientation_report = QPlainTextEdit()
        self.orientation_report.setReadOnly(True)
        self.orientation_report.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.orientation_report.setFont(
            QFontDatabase.systemFont(QFontDatabase.FixedFont)
        )
        self.orientation_report.setPlaceholderText(
            'Press "Find Grain Orientation" to calculate the relationship.'
        )
        self.tabs.addTab(self.orientation_report, 'Orientation relationship')

        csl_widget = QWidget()
        csl_layout = QVBoxLayout(csl_widget)
        self.csl_summary = QLabel(
            'CSL results will appear after the orientation is calculated.'
        )
        self.csl_summary.setWordWrap(True)
        self.csl_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        csl_layout.addWidget(self.csl_summary)

        self.csl_table = QTableWidget(0, 7)
        self.csl_table.setHorizontalHeaderLabels(
            [
                'Rank',
                'CSL (internal suffix)',
                'Ideal angle',
                'Ideal axis family',
                'Full-rotation deviation',
                'Brandon limit',
                'Inside limit?',
            ]
        )
        self.csl_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.csl_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.csl_table.verticalHeader().setVisible(False)
        header = self.csl_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        csl_layout.addWidget(self.csl_table, 1)

        caution = QLabel(
            '<b>Scope:</b> Σ describes the 3-D lattice misorientation only. '
            'Both grains must represent the same cubic phase and parent lattice; '
            'the module does not verify phase identity. '
            'The grain-boundary plane is not measured here, so a CSL match alone '
            'does not establish a coherent twin or boundary-plane character. '
            'Letter suffixes are deterministic internal variant identifiers and '
            'are not guaranteed to match every published CSL table.'
        )
        caution.setWordWrap(True)
        csl_layout.addWidget(caution)
        self.tabs.addTab(csl_widget, 'Cubic CSL checker')

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _connect_signals(self):
        self.find_button.clicked.connect(self.calculate)
        self.grain_1.currentIndexChanged.connect(self._prevent_same_grain)
        self.grain_2.currentIndexChanged.connect(self._prevent_same_grain)
        self.grain_1.currentIndexChanged.connect(self._inputs_changed)
        self.grain_2.currentIndexChanged.connect(self._inputs_changed)
        self.use_current_scan.toggled.connect(self._inputs_changed)
        self.max_sigma.valueChanged.connect(self._inputs_changed)
        self.brandon_constant.valueChanged.connect(self._inputs_changed)
        self.reflections_editor.reflections_data_changed.connect(
            self._on_reflections_changed
        )

    @property
    def reflections(self) -> ExternalReflections | None:
        return self.reflections_editor.reflections

    @property
    def scan_num(self) -> int:
        return self.reflections_editor.frame_tracker.scan_num

    def _grain_label(self, crystal_id: int) -> str:
        reflections = self.reflections
        if reflections is None:
            return f'Grain {crystal_id}'

        names = reflections.crystal_names
        if crystal_id >= len(names):
            return f'Grain {crystal_id}'
        name = names[crystal_id]
        if isinstance(name, bytes):
            name = name.decode(errors='replace')
        return f'Grain {crystal_id} — {name}' if name else f'Grain {crystal_id}'

    def _populate_grains(self):
        selected_1 = self.grain_1.currentData()
        selected_2 = self.grain_2.currentData()
        self.grain_1.blockSignals(True)
        self.grain_2.blockSignals(True)
        self.grain_1.clear()
        self.grain_2.clear()
        reflections = self.reflections
        num_crystals = 0 if reflections is None else reflections.num_crystals
        for crystal_id in range(num_crystals):
            label = self._grain_label(crystal_id)
            self.grain_1.addItem(label, crystal_id)
            self.grain_2.addItem(label, crystal_id)

        if num_crystals >= 2:
            index_1 = self.grain_1.findData(selected_1)
            index_2 = self.grain_2.findData(selected_2)
            self.grain_1.setCurrentIndex(index_1 if index_1 >= 0 else 0)
            self.grain_2.setCurrentIndex(index_2 if index_2 >= 0 else 1)
            if self.grain_1.currentData() == self.grain_2.currentData():
                self.grain_2.setCurrentIndex(
                    1 if self.grain_1.currentIndex() == 0 else 0
                )
        elif num_crystals == 1:
            self.grain_1.setCurrentIndex(0)
            self.grain_2.setCurrentIndex(0)
        self.grain_1.blockSignals(False)
        self.grain_2.blockSignals(False)
        self.find_button.setEnabled(num_crystals >= 2)
        self._prevent_same_grain()

    def _clear_results(self, message: str):
        self.orientation_report.clear()
        self.orientation_report.setPlaceholderText(message)
        self.csl_table.setRowCount(0)
        self.csl_summary.setText(message)

    def _inputs_changed(self, *args):
        if self.orientation_report.toPlainText() or self.csl_table.rowCount():
            self._clear_results(
                'Inputs changed. Press "Find Grain Orientation" to recalculate.'
            )

    def _on_reflections_changed(self):
        self._populate_grains()
        self._clear_results(
            'The reflections data changed. Select the grains and recalculate.'
        )

    def on_scan_changed(self):
        """Invalidate a tracked-orientation result after the scan changes."""

        if self.use_current_scan.isChecked() and (
            self.orientation_report.toPlainText() or self.csl_table.rowCount()
        ):
            self._clear_results(
                'The displayed scan changed. Recalculate the tracked orientations.'
            )

    def _prevent_same_grain(self):
        # Do not silently change the user's choice.  Give immediate visual
        # feedback and enforce distinct IDs when Calculate is pressed.
        same = (
            self.grain_1.currentData() is not None
            and self.grain_1.currentData() == self.grain_2.currentData()
        )
        self.find_button.setToolTip('Choose two different grains.' if same else '')

    def _select_abc_matrix(self, crystal_id: int) -> _SelectedMatrix:
        reflections = self.reflections
        if reflections is None:
            raise ValueError('No reflections file is loaded')
        if crystal_id < 0 or crystal_id >= reflections.num_crystals:
            raise ValueError(
                f'Grain {crystal_id} is no longer present in the reflections file. '
                'Select the grains again.'
            )

        original = reflections.crystals_table[crystal_id]
        indexed_scan = int(reflections.crystal_scan_number(crystal_id))
        current_scan = int(self.scan_num)
        indexed_scan_text = (
            f'scan {indexed_scan}' if indexed_scan > 0 else 'an unknown scan'
        )

        if not self.use_current_scan.isChecked():
            return _SelectedMatrix(
                original,
                f'stored ABC matrix (indexed at {indexed_scan_text})',
            )

        if indexed_scan > 0 and indexed_scan == current_scan:
            return _SelectedMatrix(
                original,
                f'stored ABC matrix indexed at displayed scan {current_scan}',
            )

        angular_shift = reflections.angular_shift_matrix(crystal_id, current_scan)
        if angular_shift is not None:
            return _SelectedMatrix(
                apply_angular_shift(original, angular_shift),
                f'tracked orientation at displayed scan {current_scan}',
            )

        warning = (
            f'Grain {crystal_id} has no tracked orientation at displayed scan '
            f'{current_scan}; its stored ABC matrix indexed at '
            f'{indexed_scan_text} was used.'
        )
        return _SelectedMatrix(
            original,
            f'stored ABC matrix (indexed at {indexed_scan_text})',
            warning,
        )

    @staticmethod
    def _format_vector(vector: np.ndarray) -> str:
        if np.any(~np.isfinite(vector)):
            return 'not unique (zero disorientation)'
        return '[' + '  '.join(f'{value: .7f}' for value in vector) + ']'

    @staticmethod
    def _format_direction(direction: tuple[int, int, int] | None) -> str:
        if direction is None:
            return 'not unique'
        return '[' + ' '.join(str(value) for value in direction) + ']'

    @staticmethod
    def _format_direction_family(direction: tuple[int, int, int] | None) -> str:
        if direction is None:
            return 'not unique'
        return '⟨' + ' '.join(str(value) for value in direction) + '⟩'

    @staticmethod
    def _format_matrix(matrix: np.ndarray) -> str:
        return np.array2string(
            matrix,
            formatter={'float_kind': lambda value: f'{value: .8f}'},
        )

    def _orientation_text(
        self,
        relationship: OrientationRelationship,
        crystal_id_1: int,
        crystal_id_2: int,
        selected_1: _SelectedMatrix,
        selected_2: _SelectedMatrix,
    ) -> str:
        orthogonality_1, determinant_1 = rotation_matrix_residuals(
            relationship.orientation_1
        )
        orthogonality_2, determinant_2 = rotation_matrix_residuals(
            relationship.orientation_2
        )
        lattice_parameter_1 = float(
            np.linalg.norm(np.asarray(selected_1.abc).reshape(3, 3)[0])
        )
        lattice_parameter_2 = float(
            np.linalg.norm(np.asarray(selected_2.abc).reshape(3, 3)[0])
        )
        lines = [
            'CUBIC GRAIN ORIENTATION RELATIONSHIP',
            '=====================================',
            f'Grain 1: {self._grain_label(crystal_id_1)}',
            f'  source: {selected_1.source}',
            f'Grain 2: {self._grain_label(crystal_id_2)}',
            f'  source: {selected_2.source}',
            '',
        ]
        warnings = [x for x in (selected_1.warning, selected_2.warning) if x]
        if warnings:
            lines.append('WARNING')
            lines.extend(f'  {warning}' for warning in warnings)
            lines.append(
                '  The result therefore compares orientations from the stated '
                'scan sources, not necessarily the same scan.'
            )
            lines.append('')

        lines.extend(
            [
                f'Raw relative rotation angle:        '
                f'{relationship.raw_angle_deg:.8f}°',
                f'Cubic minimum disorientation angle: '
                f'{relationship.angle_deg:.8f}°',
                '',
                'DIRECT FIXED-CELL ORIENTATION VALIDATION',
                '----------------------------------------',
                f'Grain {crystal_id_1}: a={lattice_parameter_1:.8g}, '
                f'max|U^T U-I|={orthogonality_1:.3e}, '
                f'|det(U)-1|={determinant_1:.3e}',
                f'Grain {crystal_id_2}: a={lattice_parameter_2:.8g}, '
                f'max|U^T U-I|={orthogonality_2:.3e}, '
                f'|det(U)-1|={determinant_2:.3e}',
                '',
                'DISORIENTATION AXIS',
                '-------------------',
                f'Laboratory coordinates: {self._format_vector(relationship.axis_lab)}',
                f'Grain {crystal_id_1} crystal coordinates: '
                f'{self._format_vector(relationship.axis_crystal_1)}',
                f'Grain {crystal_id_2} crystal coordinates: '
                f'{self._format_vector(relationship.axis_crystal_2)}',
                '',
                'Nearest primitive low-index directions (|index| <= 6):',
                f'  Grain {crystal_id_1}: '
                f'{self._format_direction(relationship.axis_uvw_1)}  '
                f'(angular residual {relationship.axis_uvw_error_deg_1:.6f}°)',
                f'  Grain {crystal_id_2}: '
                f'{self._format_direction(relationship.axis_uvw_2)}  '
                f'(angular residual {relationship.axis_uvw_error_deg_2:.6f}°)',
                '',
            ]
        )

        if relationship.axis_uvw_1 is not None:
            lines.extend(
                [
                    'CRYSTALLOGRAPHIC AXIS RELATIONSHIP',
                    '----------------------------------',
                    f'{self._format_direction(relationship.axis_uvw_1)}'
                    f' (grain {crystal_id_1})  ∥  '
                    f'{self._format_direction(relationship.axis_uvw_2)}'
                    f' (grain {crystal_id_2})',
                    f'with a cubic disorientation of {relationship.angle_deg:.8f}°.',
                    'The integer directions are approximations unless their '
                    'reported residuals are zero; the vectors above are the '
                    'calculated axes.',
                    '',
                ]
            )

        lines.extend(
            [
                'DISORIENTATION MATRIX',
                '---------------------',
                'Maps symmetry-equivalent grain-2 cubic coordinates into '
                'grain-1 cubic coordinates:',
                self._format_matrix(relationship.disorientation),
                '',
                'RAW MISORIENTATION MATRIX',
                '-------------------------',
                'M = U1^T U2 (before cubic symmetry reduction):',
                self._format_matrix(relationship.raw_misorientation),
                '',
                'Method: for each fixed cubic ABC matrix C, the orientation '
                'is calculated directly as U = C^T/a. U^T U = I and '
                'det(U) = +1 are validated; no matrix projection or metric '
                'refinement is applied. All 24 x 24 proper cubic symmetry '
                'combinations are then tested.',
                'Exact CSL references use Ranganathan axis–angle parameters '
                'and the explicit rational 3 x 3 rotation matrix. Sigma is '
                'its reduced common denominator.',
                'Assumption: both selected grains represent the same cubic '
                'phase and parent lattice. Phase identity and boundary-plane '
                'geometry are not validated by this calculation.',
                'CSL references: Ranganathan, Acta Cryst. 21 (1966) '
                '197–199; Grimmer, Bollmann & Warrington, Acta Cryst. A30 '
                '(1974) 197–207. Brandon acceptance: Acta Metall. 14 '
                '(1966) 1479–1484.',
            ]
        )
        return '\n'.join(lines)

    def _populate_csl_results(self, relationship: OrientationRelationship):
        matches = match_cubic_csls(
            relationship.raw_misorientation,
            max_sigma=self.max_sigma.value(),
            brandon_constant_deg=self.brandon_constant.value(),
        )
        displayed = matches[: min(12, len(matches))]
        self.csl_table.setRowCount(len(displayed))

        for row, match in enumerate(displayed):
            values = [
                str(row + 1),
                match.csl.label,
                f'{match.csl.angle_deg:.6f}°',
                self._format_direction_family(match.csl.axis_uvw),
                f'{match.deviation_deg:.6f}°',
                f'{match.brandon_limit_deg:.6f}°',
                'yes' if match.within_brandon_limit else 'no',
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 2, 4, 5, 6):
                    item.setTextAlignment(Qt.AlignCenter)
                self.csl_table.setItem(row, column, item)

        if not matches:
            self.csl_summary.setText('No CSL rotations were generated.')
            return

        nearest = matches[0]
        accepted = [match for match in matches if match.within_brandon_limit]
        if accepted:
            best_normalized = min(
                accepted,
                key=lambda match: (
                    match.normalized_deviation,
                    match.csl.sigma,
                ),
            )
            acceptance = (
                f'Best Brandon-normalized accepted match: '
                f'<b>{best_normalized.csl.label}</b> '
                f'(Δ={best_normalized.deviation_deg:.4f}°, '
                f'limit={best_normalized.brandon_limit_deg:.4f}°).'
            )
        else:
            acceptance = 'No generated CSL lies inside the selected Brandon limit.'

        self.csl_summary.setText(
            f'Nearest exact CSL in full rotation space: '
            f'<b>{nearest.csl.label}</b>, ideal '
            f'{nearest.csl.angle_deg:.4f}° about '
            f'{self._format_direction_family(nearest.csl.axis_uvw)}, '
            f'Δ={nearest.deviation_deg:.4f}°. {acceptance} '
            f'The table shows the 12 nearest exact CSL rotations through '
            f'Σ≤{self.max_sigma.value()}. Letter suffixes are internal variant '
            f'identifiers; use Σ, ideal angle, and axis together when comparing '
            f'with the literature.'
        )

    def calculate(self):
        crystal_id_1 = self.grain_1.currentData()
        crystal_id_2 = self.grain_2.currentData()
        if crystal_id_1 is None or crystal_id_2 is None:
            QMessageBox.warning(
                self,
                'Not Enough Grains',
                'At least two indexed grains are required.',
            )
            return
        if crystal_id_1 == crystal_id_2:
            QMessageBox.warning(
                self,
                'Choose Different Grains',
                'Grain 1 and Grain 2 must be different indexed grains.',
            )
            return

        try:
            selected_1 = self._select_abc_matrix(int(crystal_id_1))
            selected_2 = self._select_abc_matrix(int(crystal_id_2))
            relationship = analyze_orientation_relationship(
                selected_1.abc,
                selected_2.abc,
            )
            report = self._orientation_text(
                relationship,
                int(crystal_id_1),
                int(crystal_id_2),
                selected_1,
                selected_2,
            )
            self.orientation_report.setPlainText(report)
            self._populate_csl_results(relationship)
        except (
            IndexError,
            KeyError,
            OSError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            QMessageBox.critical(
                self,
                'Orientation Calculation Failed',
                str(exc),
            )
