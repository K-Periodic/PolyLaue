# Copyright © 2026, UChicago Argonne, LLC. See "LICENSE" for full details.

import numpy as np
import pytest

from polylaue.model.core import (
    abc_to_orientation,
    analyze_orientation_relationship,
    axis_angle_rotation_matrix,
    cubic_csl_rotation_matrix,
    generate_cubic_csls,
    match_cubic_csls,
    proper_cubic_symmetry_operators,
)


def _abc_from_orientation(
    orientation: np.ndarray,
    lattice_parameter: float = 5.0,
) -> np.ndarray:
    # PolyLaue ABC vectors are rows, while orientation maps crystal columns
    # into laboratory coordinates.
    return (lattice_parameter * orientation.T).ravel()


def _axis_rotation(angle_deg: float, axis: tuple[int, int, int]) -> np.ndarray:
    return axis_angle_rotation_matrix(axis, angle_deg)


def test_proper_cubic_symmetry_group():
    operators = proper_cubic_symmetry_operators()
    assert operators.shape == (24, 3, 3)
    assert len({tuple(operator.ravel()) for operator in operators}) == 24
    assert np.allclose(
        np.matmul(operators, operators.transpose(0, 2, 1)),
        np.eye(3),
    )
    assert np.allclose(np.linalg.det(operators), 1.0)


def test_fixed_cubic_abc_gives_orientation_directly():
    orientation = _axis_rotation(31.0, (1, 2, 3))
    abc = _abc_from_orientation(orientation, lattice_parameter=5.2)
    assert np.allclose(abc_to_orientation(abc), orientation, atol=1e-12)


def test_non_cubic_metric_is_rejected_instead_of_projected():
    orientation = _axis_rotation(31.0, (1, 2, 3))
    symmetric_stretch = np.array(
        [
            [5.2, 0.08, -0.03],
            [0.08, 4.8, 0.05],
            [-0.03, 0.05, 5.1],
        ]
    )
    abc = (symmetric_stretch @ orientation.T).ravel()
    with pytest.raises(ValueError, match='No projection or correction was applied'):
        abc_to_orientation(abc)


def test_cubic_symmetry_reduces_a_fourfold_rotation_to_zero():
    orientation_1 = np.eye(3)
    orientation_2 = _axis_rotation(90.0, (0, 0, 1))
    relationship = analyze_orientation_relationship(
        _abc_from_orientation(orientation_1),
        _abc_from_orientation(orientation_2),
    )

    assert relationship.raw_angle_deg == pytest.approx(90.0, abs=1e-10)
    assert relationship.angle_deg == pytest.approx(0.0, abs=1e-10)
    assert np.all(np.isnan(relationship.axis_lab))


def test_ranganathan_sigma_5_is_an_explicit_rational_rotation_matrix():
    rotation, sigma, angle_deg = cubic_csl_rotation_matrix(
        axis_uvw=(1, 0, 0),
        x=3,
        y=1,
    )

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 4.0 / 5.0, -3.0 / 5.0],
            [0.0, 3.0 / 5.0, 4.0 / 5.0],
        ]
    )
    assert np.allclose(rotation, expected)
    assert sigma == 5
    assert angle_deg == pytest.approx(36.8698976458, abs=1e-10)


def test_sigma_3_relationship_reports_axis_in_both_grains():
    orientation_1 = _axis_rotation(17.0, (2, 3, 1))
    sigma_3 = _axis_rotation(60.0, (1, 1, 1))
    orientation_2 = orientation_1 @ sigma_3
    relationship = analyze_orientation_relationship(
        _abc_from_orientation(orientation_1),
        _abc_from_orientation(orientation_2),
    )

    assert relationship.raw_angle_deg == pytest.approx(60.0, abs=1e-10)
    assert relationship.angle_deg == pytest.approx(60.0, abs=1e-10)
    assert relationship.axis_uvw_1 == (1, 1, 1)
    assert relationship.axis_uvw_2 == (1, 1, 1)
    assert relationship.axis_uvw_error_deg_1 == pytest.approx(0.0, abs=1e-8)
    assert relationship.axis_uvw_error_deg_2 == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    ('label', 'angle_deg', 'axis'),
    [
        ('Σ3', 60.0, (1, 1, 1)),
        ('Σ5', 36.8698976458, (1, 0, 0)),
        ('Σ7', 38.2132107017, (1, 1, 1)),
        ('Σ9', 38.9424412689, (1, 1, 0)),
        ('Σ11', 50.4788036414, (1, 1, 0)),
        ('Σ13a', 22.6198649480, (1, 0, 0)),
        ('Σ13b', 27.7957724960, (1, 1, 1)),
        ('Σ27a', 31.5863380965, (1, 1, 0)),
        ('Σ27b', 35.4309446873, (2, 1, 0)),
    ],
)
def test_generated_low_sigma_csl_table(label, angle_deg, axis):
    table = {csl.label: csl for csl in generate_cubic_csls(29)}
    assert label in table
    assert table[label].angle_deg == pytest.approx(angle_deg, abs=1e-8)
    assert table[label].axis_uvw == axis


def test_generated_csl_catalogue_deduplicates_grain_exchange():
    catalogue = generate_cubic_csls(49)
    sigma_39 = [csl for csl in catalogue if csl.sigma == 39]

    # The conventional cubic catalogue through Sigma 49 has 48 unoriented
    # classes including Sigma 1.  A symmetry-only reduction incorrectly
    # retained a third Sigma-39 entry related to Sigma-39b by R -> R^T.
    assert len(catalogue) == 48
    assert [csl.label for csl in sigma_39] == ['Σ39a', 'Σ39b']

    # These two exact Ranganathan rotations generated the old duplicate.  Both
    # must now resolve to the same single unoriented catalogue entry.
    for x in (5, 8):
        rotation, sigma, _ = cubic_csl_rotation_matrix((3, 2, 1), x=x, y=1)
        assert sigma == 39

        matches = match_cubic_csls(rotation, max_sigma=49)
        exact_sigma_39 = [
            match
            for match in matches
            if match.csl.sigma == 39 and match.deviation_deg < 1e-8
        ]
        assert [match.csl.label for match in exact_sigma_39] == ['Σ39b']


def test_cached_csl_catalogue_arrays_are_read_only():
    csl = generate_cubic_csls(5)[-1]

    assert not csl.rotation.flags.writeable
    assert not csl.axis_crystal.flags.writeable
    with pytest.raises(ValueError, match='read-only'):
        csl.rotation[0, 0] = 0.0
    with pytest.raises(ValueError, match='read-only'):
        csl.axis_crystal[0] = 0.0


def test_full_rotation_csl_match_recovers_exact_sigma_5():
    sigma_5 = _axis_rotation(36.8698976458, (1, 0, 0))
    matches = match_cubic_csls(sigma_5, max_sigma=29)

    assert matches[0].csl.label == 'Σ5'
    assert matches[0].deviation_deg == pytest.approx(0.0, abs=1e-8)
    assert matches[0].within_brandon_limit


def test_csl_match_does_not_use_angle_alone():
    # This has the Sigma-5 angle but a non-Sigma-5 axis.  An angle-only
    # lookup would incorrectly report zero deviation.
    wrong_axis = _axis_rotation(36.8698976458, (1, 2, 3))
    matches = match_cubic_csls(wrong_axis, max_sigma=5)
    sigma_5_match = next(match for match in matches if match.csl.label == 'Σ5')

    assert sigma_5_match.deviation_deg > 1.0
