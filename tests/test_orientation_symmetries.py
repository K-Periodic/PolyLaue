# Copyright c 2026, UChicago Argonne, LLC. See "LICENSE" for full details.

"""Numerical checks that require only NumPy and the Python standard library.

Run from the source root:
    python -m unittest discover -s tests -p test_orientation_symmetries.py -v

Load the pure numerical module directly so these checks do not require Qt,
image readers, or the unrelated indexing dependencies.
"""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_path = (
    Path(__file__).resolve().parents[1]
    / 'polylaue/model/core/orientation_relationship.py'
)
_spec = importlib.util.spec_from_file_location('_polylaue_orientation_checks', _path)
orientation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = orientation
_spec.loader.exec_module(orientation)


def reference_basis(parameters):
    """Independent fixed-cell construction using find()'s axis convention."""
    a, b, c, alpha, beta, gamma = parameters
    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])
    bz, by = b * np.cos(alpha), b * np.sin(alpha)
    az = a * np.cos(beta)
    ay = (a * b * np.cos(gamma) - az * bz) / by
    ax = np.sqrt(a * a - ay * ay - az * az)
    return np.array([[ax, 0, 0], [ay, by, 0], [az, bz, c]])


CASES = [
    ('m-3m', 'standard', (3, 3, 3, 90, 90, 90), 24),
    ('m-3', 'standard', (3, 3, 3, 90, 90, 90), 12),
    ('6/mmm', 'standard', (3, 3, 5, 90, 90, 120), 12),
    ('6/m', 'standard', (3, 3, 5, 90, 90, 120), 6),
    ('-3m', 'hexagonal-a', (3, 3, 5, 90, 90, 120), 6),
    ('-3m', 'hexagonal-a-b', (3, 3, 5, 90, 90, 120), 6),
    ('-3m', 'rhombohedral', (3, 3, 3, 75, 75, 75), 6),
    ('-3', 'hexagonal', (3, 3, 5, 90, 90, 120), 3),
    ('-3', 'rhombohedral', (3, 3, 3, 75, 75, 75), 3),
    ('4/mmm', 'standard', (3, 3, 5, 90, 90, 90), 8),
    ('4/m', 'standard', (3, 3, 5, 90, 90, 90), 4),
    ('mmm', 'standard', (3, 4, 5, 90, 90, 90), 4),
    ('2/m', 'unique-b', (3, 4, 5, 90, 105, 90), 2),
    ('2/m', 'unique-a', (3, 4, 5, 105, 90, 90), 2),
    ('2/m', 'unique-c', (3, 4, 5, 90, 90, 105), 2),
    ('-1', 'standard', (3, 4, 5, 80, 95, 110), 1),
]


class TestOrientationSymmetries(unittest.TestCase):
    def test_fixed_cell_orientation_recovery_and_metric_preservation(self):
        U = orientation.axis_angle_rotation_matrix((1, 2, 3), 31)
        for laue, setting, parameters, _ in CASES:
            with self.subTest(laue=laue, setting=setting):
                A_0 = reference_basis(parameters)
                C = (U @ A_0).T
                original = C.copy()
                recovered = orientation.abc_to_orientation(
                    C, laue_class=laue, setting=setting
                )
                np.testing.assert_allclose(recovered, U, atol=1e-12)
                reconstructed = orientation.fixed_reference_basis(C)
                np.testing.assert_allclose(reconstructed, A_0, atol=1e-12)
                np.testing.assert_allclose(
                    C @ C.T, reconstructed.T @ reconstructed, atol=1e-12
                )
                np.testing.assert_array_equal(C, original)

    def test_all_symmetry_groups_are_proper_unique_and_closed(self):
        for laue, setting, parameters, count in CASES:
            with self.subTest(laue=laue, setting=setting):
                operators = orientation.proper_crystal_symmetry_operators(
                    reference_basis(parameters), laue, setting
                )
                self.assertEqual(len(operators), count)
                unique = {tuple(np.round(S.ravel(), 10)) for S in operators}
                self.assertEqual(len(unique), count)
                np.testing.assert_allclose(np.linalg.det(operators), 1, atol=1e-12)
                np.testing.assert_allclose(operators[0], np.eye(3), atol=1e-12)
                for S in operators:
                    np.testing.assert_allclose(S.T @ S, np.eye(3), atol=1e-12)
                    for T in operators:
                        distances = np.max(np.abs(operators - S @ T), axis=(1, 2))
                        self.assertLess(float(np.min(distances)), 1e-12)

    def test_symmetry_equivalent_grains_have_zero_disorientation(self):
        U = orientation.axis_angle_rotation_matrix((2, 3, 1), 17)
        for laue, setting, parameters, _ in CASES:
            A_0 = reference_basis(parameters)
            for S in orientation.proper_crystal_symmetry_operators(A_0, laue, setting):
                with self.subTest(laue=laue, setting=setting, operation=S):
                    result = orientation.analyze_orientation_relationship(
                        (U @ A_0).T, (U @ S @ A_0).T,
                        laue_class=laue, setting=setting,
                    )
                    self.assertAlmostEqual(result.angle_deg, 0, places=8)
                    self.assertTrue(np.all(np.isnan(result.axis_lab)))

    def test_rotation_axes_agree_in_both_grains_and_lab(self):
        U_1 = orientation.axis_angle_rotation_matrix((2, 3, 1), 17)
        for laue, setting, parameters, _ in CASES:
            with self.subTest(laue=laue, setting=setting):
                A_0 = reference_basis(parameters)
                axis = A_0 @ np.array([1, 2, 1])
                axis /= np.linalg.norm(axis)
                U_2 = U_1 @ orientation.axis_angle_rotation_matrix(axis, 7)
                C_1, C_2 = (U_1 @ A_0).T, (U_2 @ A_0).T
                result = orientation.analyze_orientation_relationship(
                    C_1, C_2, laue_class=laue, setting=setting
                )
                self.assertAlmostEqual(result.raw_angle_deg, 7, places=8)
                self.assertAlmostEqual(result.angle_deg, 7, places=8)
                np.testing.assert_allclose(
                    U_1 @ result.axis_crystal_1, result.axis_lab, atol=1e-12
                )
                np.testing.assert_allclose(
                    U_2 @ result.axis_crystal_2, result.axis_lab, atol=1e-12
                )
                for C, indices in ((C_1, result.axis_uvw_1), (C_2, result.axis_uvw_2)):
                    indexed_axis = C.T @ np.array(indices)
                    indexed_axis /= np.linalg.norm(indexed_axis)
                    self.assertAlmostEqual(
                        abs(indexed_axis @ result.axis_lab), 1, places=12
                    )

    def test_direction_labels_use_direct_lattice_metric(self):
        for parameters in (
            (3, 3, 12, 90, 90, 90),
            (3, 3, 5, 90, 90, 120),
            (3, 4, 5, 80, 95, 110),
        ):
            with self.subTest(parameters=parameters):
                A_0 = reference_basis(parameters)
                direction, error = orientation.nearest_lattice_direction(
                    A_0 @ np.array([1, 0, 1]), reference_basis=A_0
                )
                self.assertEqual(direction, (1, 0, 1))
                self.assertLess(error, 2e-6)

    def test_lower_symmetry_is_not_replaced_by_higher_lattice_symmetry(self):
        A_0 = reference_basis((3, 3, 5, 90, 90, 120))
        U = orientation.axis_angle_rotation_matrix((0, 0, 1), 60)
        hexagonal = orientation.analyze_orientation_relationship(
            A_0.T, (U @ A_0).T, laue_class='6/m'
        )
        trigonal = orientation.analyze_orientation_relationship(
            A_0.T, (U @ A_0).T, laue_class='-3'
        )
        self.assertAlmostEqual(hexagonal.angle_deg, 0, places=8)
        self.assertAlmostEqual(trigonal.angle_deg, 60, places=8)

        A_0 = reference_basis((3, 3, 5, 90, 90, 90))
        U = orientation.axis_angle_rotation_matrix((1, 0, 0), 180)
        high = orientation.analyze_orientation_relationship(
            A_0.T, (U @ A_0).T, laue_class='4/mmm'
        )
        low = orientation.analyze_orientation_relationship(
            A_0.T, (U @ A_0).T, laue_class='4/m'
        )
        self.assertAlmostEqual(high.angle_deg, 0, places=8)
        self.assertAlmostEqual(low.angle_deg, 180, places=5)

    def test_trigonal_hexagonal_settings_select_different_twofold_axes(self):
        A_0 = reference_basis((3, 3, 5, 90, 90, 120))
        U = orientation.axis_angle_rotation_matrix(A_0[:, 0].copy(), 180)
        angles = []
        for setting in ('hexagonal-a', 'hexagonal-a-b'):
            result = orientation.analyze_orientation_relationship(
                A_0.T, (U @ A_0).T, laue_class='-3m', setting=setting
            )
            angles.append(result.angle_deg)
        self.assertAlmostEqual(angles[0], 0, places=8)
        self.assertAlmostEqual(angles[1], 60, places=8)

    def test_incompatible_cell_and_wrong_monoclinic_setting_are_rejected(self):
        C = reference_basis((3, 4, 5, 90, 90, 90)).T
        with self.assertRaisesRegex(ValueError, 'No projection or correction'):
            orientation.abc_to_orientation(C, laue_class='4/mmm')
        C = reference_basis((3, 4, 5, 90, 105, 90)).T
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            orientation.abc_to_orientation(C, laue_class='2/m', setting='unique-c')

    def test_invalid_bases_and_unknown_symmetry_are_rejected(self):
        for C in (
            np.diag([-1., 1, 1]), np.diag([1., 1, 0]), np.full((3, 3), np.nan)
        ):
            with self.subTest(C=C), self.assertRaises(ValueError):
                orientation.abc_to_orientation(C, laue_class='-1')
        for laue, setting in (('unknown', 'standard'), ('mmm', 'unique-a')):
            with self.subTest(laue=laue), self.assertRaises(ValueError):
                orientation.abc_to_orientation(
                    np.eye(3), laue_class=laue, setting=setting
                )

    def test_near_half_turn_axis_is_stable(self):
        A_0 = reference_basis((3, 4, 5, 80, 95, 110))
        axis = np.array([1., 2, 3])
        axis /= np.linalg.norm(axis)
        U = orientation.axis_angle_rotation_matrix(axis, 179.999)
        result = orientation.analyze_orientation_relationship(
            A_0.T, (U @ A_0).T, laue_class='-1'
        )
        self.assertAlmostEqual(result.angle_deg, 179.999, places=7)
        self.assertAlmostEqual(abs(result.axis_lab @ axis), 1, places=12)

    def test_existing_cubic_csl_catalogue_and_matches(self):
        catalogue = orientation.generate_cubic_csls(49)
        self.assertEqual(len(catalogue), 48)
        self.assertEqual(
            [csl.label for csl in catalogue if csl.sigma == 39],
            ['\u03a339a', '\u03a339b'],
        )
        R, sigma, angle = orientation.cubic_csl_rotation_matrix((1, 0, 0), 3, 1)
        self.assertEqual(sigma, 5)
        self.assertAlmostEqual(angle, 36.8698976458, places=9)
        match = orientation.match_cubic_csls(R, max_sigma=49)[0]
        self.assertEqual(match.csl.sigma, 5)
        self.assertLess(match.deviation_deg, 1e-8)
        self.assertTrue(match.within_brandon_limit)
        self.assertIs(orientation.CubicDisorientation, orientation.Disorientation)


if __name__ == '__main__':
    unittest.main()
