# Copyright © 2026, UChicago Argonne, LLC. See "LICENSE" for full details.

"""Fixed-cell grain-orientation relationships and cubic CSL identification.

Conventions
-----------
PolyLaue stores the direct-lattice vectors as the *rows* of the 3 x 3
``ABC`` matrix ``C``.  PolyLaue keeps the unit-cell parameters fixed, so for
a cubic cell with lattice parameter ``a`` the indexed matrix is

    C = a U^T,

where ``U`` maps a column vector in crystal coordinates into laboratory
coordinates.  Consequently the orientation is calculated directly,

    U = C^T / a.

Equivalently, if ``A_0`` and ``B_0 = A_0^-T`` are the fixed direct and
reciprocal reference bases, then

    U = C^T A_0^-1 = C^-1 B_0^-1.



For any fixed cell, A_0 is reconstructed in the same convention as find():
c parallel to Z, b in the YZ plane, and a with a positive X component.
No lattice parameters are refined. In this orthonormal crystal frame,

    C^T = U A_0,          U = C^T A_0^-1.

A crystallographic direction d = [u, v, w]^T is represented by A_0 d.
An integer fractional-coordinate symmetry W is represented by the proper
Cartesian rotation S = A_0 W A_0^-1, with W^T G W = G and G = A_0^T A_0.
The selected Laue class supplies its proper rotational subgroup. Settings
are explicit for monoclinic and trigonal cells; symmetry is not inferred
from the cell metric. See International Tables for Crystallography A
(2016), sections 1.3.2 and 1.3.4, and Glazer, Aroyo & Authier, Acta Cryst.
A70 (2014) 300--302, DOI 10.1107/S2053273314004495, for the symmetry matrices.
For direct/reciprocal coordinate transformations and angle--axis analysis
of micro-Laue data, see Li, Wan & Chen, J. Appl. Cryst. 48 (2015) 747--757,
DOI 10.1107/S1600576715004896, equations (3)--(7).

For two grains, the relative orientation is written in the conventional
crystal-coordinate form

    M = U_1^T U_2.

For the proper rotations ``S_i`` of each selected crystal symmetry, all
symmetry-equivalent relative orientations are

    M_ij = S_i^T M S_j.

The disorientation is the member with the smallest rotation angle.
The eigenvector with eigenvalue one is transformed back into the original
coordinates of each grain, so the reported crystallographic direction pair
and the laboratory rotation axis refer to the actual indexed grains.

Cubic coincidence rotations are generated with Ranganathan's axis--angle
parameters.  For a primitive cubic direction ``d = [u, v, w]`` and coprime
non-negative integers ``x`` and ``y``, define

    p = y d,        D = x^2 + p^T p,
    theta = 2 atan2(sqrt(p^T p), x).

The corresponding rotation is kept explicitly in matrix form:

    R = ((x^2 - p^T p) I + 2 p p^T + 2 x [p]_x) / D,

where ``[p]_x`` is the cross-product matrix.  After cancelling the common
integer factor in the numerator and denominator, the reduced denominator is
``Sigma``.  This is the rational rotation-matrix criterion of Grimmer,
Bollmann and Warrington.

See S. Ranganathan, Acta Cryst. 21 (1966) 197--199,
DOI 10.1107/S0365110X66002615; H. Grimmer, W. Bollmann and
D. H. Warrington, Acta Cryst. A30 (1974) 197--207,
DOI 10.1107/S056773947400043X; and H. Grimmer, Acta Cryst. A30 (1974)
685--688, DOI 10.1107/S0567739474001719.  The optional acceptance limit is
the Brandon criterion, Delta-theta = 15 degrees / sqrt(Sigma), from
D. G. Brandon, Acta Metall. 14 (1966) 1479--1484,
DOI 10.1016/0001-6160(66)90168-4.

The CSL result classifies only the three-dimensional lattice
misorientation.  It does not determine the grain-boundary plane and cannot,
by itself, establish that a boundary is a coherent twin.
"""

from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
import string

import numpy as np


LAUE_CLASSES = (
    ('m-3m', 'Cubic m-3m'),
    ('m-3', 'Cubic m-3'),
    ('6/mmm', 'Hexagonal 6/mmm'),
    ('6/m', 'Hexagonal 6/m'),
    ('-3m', 'Trigonal -3m'),
    ('-3', 'Trigonal -3'),
    ('4/mmm', 'Tetragonal 4/mmm'),
    ('4/m', 'Tetragonal 4/m'),
    ('mmm', 'Orthorhombic mmm'),
    ('2/m', 'Monoclinic 2/m'),
    ('-1', 'Triclinic -1'),
)


def laue_class_settings(laue_class: str) -> tuple[tuple[str, str], ...]:
    """Return the supported conventional axis settings for a Laue class."""

    if laue_class not in dict(LAUE_CLASSES):
        raise ValueError(f'Unknown Laue class: {laue_class}')
    if laue_class == '2/m':
        return (
            ('unique-b', 'Unique axis b'),
            ('unique-a', 'Unique axis a'),
            ('unique-c', 'Unique axis c'),
        )
    if laue_class == '-3m':
        return (
            ('hexagonal-a', 'Hexagonal axes; twofold parallel to [100]'),
            ('hexagonal-a-b', 'Hexagonal axes; twofold parallel to [1 -1 0]'),
            ('rhombohedral', 'Rhombohedral axes; threefold parallel to [111]'),
        )
    if laue_class == '-3':
        return (
            ('hexagonal', 'Hexagonal axes; threefold parallel to [001]'),
            ('rhombohedral', 'Rhombohedral axes; threefold parallel to [111]'),
        )
    return (('standard', 'Conventional axes'),)


@dataclass(frozen=True)
class Disorientation:
    """A minimum-angle representative of a symmetry-equivalent rotation."""

    rotation: np.ndarray
    angle_deg: float
    symmetry_1: np.ndarray
    symmetry_2: np.ndarray


# Preserve the existing cubic API used by the CSL catalogue.
CubicDisorientation = Disorientation


@dataclass(frozen=True)
class OrientationRelationship:
    """The raw and symmetry-reduced relationship of two same-phase grains.

    axis_crystal_1 and axis_crystal_2 are Cartesian crystal-frame vectors.
    axis_uvw_1 and axis_uvw_2 are approximate direct-lattice indices.
    """

    orientation_1: np.ndarray
    orientation_2: np.ndarray
    raw_misorientation: np.ndarray
    raw_angle_deg: float
    disorientation: np.ndarray
    angle_deg: float
    axis_lab: np.ndarray
    axis_crystal_1: np.ndarray
    axis_crystal_2: np.ndarray
    axis_uvw_1: tuple[int, int, int] | None
    axis_uvw_2: tuple[int, int, int] | None
    axis_uvw_error_deg_1: float
    axis_uvw_error_deg_2: float
    symmetry_1: np.ndarray
    symmetry_2: np.ndarray
    laue_class: str = 'm-3m'
    setting: str = 'standard'


@dataclass(frozen=True)
class CubicCsl:
    """One symmetry-distinct exact cubic coincidence rotation.

    ``variant`` is a deterministic internal suffix assigned within one Sigma.
    It is not asserted to reproduce the letter suffix used by every published
    CSL table; ``sigma``, ``angle_deg``, and ``axis_uvw`` together are the
    unambiguous identifier.
    """

    sigma: int
    variant: str
    rotation: np.ndarray
    angle_deg: float
    axis_crystal: np.ndarray
    axis_uvw: tuple[int, int, int] | None

    @property
    def label(self) -> str:
        return f'Σ{self.sigma}{self.variant}'


@dataclass(frozen=True)
class CubicCslMatch:
    """Distance from a measured misorientation to an exact cubic CSL."""

    csl: CubicCsl
    deviation_deg: float
    brandon_limit_deg: float

    @property
    def normalized_deviation(self) -> float:
        if self.brandon_limit_deg == 0:
            return math.inf
        return self.deviation_deg / self.brandon_limit_deg

    @property
    def within_brandon_limit(self) -> bool:
        return self.deviation_deg <= self.brandon_limit_deg


@lru_cache(maxsize=1)
def proper_cubic_symmetry_operators() -> np.ndarray:
    """Return the 24 proper rotational symmetries of a cubic lattice."""

    operators = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3, dtype=int)[:, permutation]
        for signs in itertools.product((-1, 1), repeat=3):
            operator = permutation_matrix @ np.diag(signs)
            if round(np.linalg.det(operator)) == 1:
                operators.append(operator.astype(float))

    # Put the identity first and otherwise use a deterministic ordering.
    operators.sort(
        key=lambda x: (
            0 if np.array_equal(x, np.eye(3)) else 1,
            tuple(x.ravel()),
        )
    )
    array = np.asarray(operators)
    array.setflags(write=False)
    return array


@lru_cache(maxsize=None)
def _fractional_symmetry_operators(laue_class: str, setting: str) -> np.ndarray:
    """Generate proper symmetries W in direct-lattice coordinates.

    The matrices below are the coordinate-triplet operations of
    International Tables A (Glazer, Aroyo & Authier, 2014, Tables 1--3).
    Cyclic groups are {W_n^k}; dihedral groups are
    {W_n^k, W_n^k W_2}, k = 0, ..., n - 1.
    """

    settings = dict(laue_class_settings(laue_class))
    if setting == 'standard':
        setting = next(iter(settings))
    if setting not in settings:
        raise ValueError(f'Unsupported setting {setting!r} for {laue_class}')

    if laue_class == 'm-3m':
        return proper_cubic_symmetry_operators()
    if laue_class == 'm-3':
        # 23: even permutations, with an even number of sign reversals.
        operators = np.array([
            W for W in proper_cubic_symmetry_operators()
            if round(np.linalg.det(np.abs(W))) == 1
        ])
    else:
        W_2a = np.diag([1, -1, -1])
        W_2b = np.diag([-1, 1, -1])
        W_2c = np.diag([-1, -1, 1])
        W_4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        W_6 = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
        W_3h = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
        W_3r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        W_2h_a = np.array([[1, -1, 0], [0, -1, 0], [0, 0, -1]])
        W_2h_a_b = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

        W_n, n, W_2 = np.eye(3, dtype=int), 1, None
        if laue_class == '2/m':
            W_n = {'unique-a': W_2a, 'unique-b': W_2b, 'unique-c': W_2c}[setting]
            n = 2
        elif laue_class == 'mmm':
            W_n, n, W_2 = W_2c, 2, W_2a
        elif laue_class in ('4/m', '4/mmm'):
            W_n, n = W_4, 4
            if laue_class == '4/mmm':
                W_2 = W_2a
        elif laue_class in ('6/m', '6/mmm'):
            W_n, n = W_6, 6
            if laue_class == '6/mmm':
                W_2 = W_2h_a
        elif laue_class in ('-3', '-3m'):
            W_n, n = (W_3r if setting == 'rhombohedral' else W_3h), 3
            if laue_class == '-3m':
                W_2 = W_2h_a if setting == 'hexagonal-a' else W_2h_a_b

        cyclic = [np.linalg.matrix_power(W_n, k) for k in range(n)]
        operators = np.array(
            cyclic if W_2 is None else cyclic + [W @ W_2 for W in cyclic]
        )
    operators.setflags(write=False)
    return operators


def fixed_reference_basis(abc_matrix: np.ndarray) -> np.ndarray:
    """Reconstruct A_0 without changing the stored fixed-cell metric.

    C contains direct vectors a, b, c as rows. find() defines its reference
    cell with c || Z and b in the YZ plane. Its oriented orthonormal frame is

        e_z = c / |c|,  e_x = (b x c) / |b x c|,  e_y = e_z x e_x,
        U = [e_x e_y e_z],  A_0 = U^T C^T,  C C^T = A_0^T A_0.

    This is a coordinate construction, not a strain fit or a projection
    onto a higher-symmetry cell. a must have a positive component along e_x.
    """

    C = np.asarray(abc_matrix, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(C)):
        raise ValueError('The ABC matrix contains a non-finite value')
    lengths = np.linalg.norm(C, axis=1)
    if np.any(lengths <= 0) or np.linalg.det(C) <= 1e-12 * np.prod(lengths):
        raise ValueError('The ABC basis must be non-degenerate and right-handed')
    e_z = C[2] / lengths[2]
    e_x = np.cross(C[1], C[2])
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    U = np.column_stack((e_x, e_y, e_z))
    return U.T @ C.T


def proper_crystal_symmetry_operators(
    reference_basis: np.ndarray,
    laue_class: str = 'm-3m',
    setting: str = 'standard',
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return S = A_0 W A_0^-1 in the Cartesian crystal reference frame.

    W^T G W = G, G = A_0^T A_0, is checked before any operator is used.
    An incompatible metric is rejected; it is never symmetrized.
    """

    A_0 = np.asarray(reference_basis, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(A_0)) or np.linalg.det(A_0) <= 0:
        raise ValueError('The reference basis must be finite and right-handed')
    G = A_0.T @ A_0
    G /= np.max(np.abs(G))
    A_0_inverse = np.linalg.inv(A_0)
    operators = []
    for W in _fractional_symmetry_operators(laue_class, setting):
        residual = float(np.max(np.abs(W.T @ G @ W - G)))
        if residual > tolerance:
            raise ValueError(
                f'The fixed cell is incompatible with Laue class {laue_class} '
                f'and setting {setting}: max|W^T G W-G|/max|G|={residual:.3e}. '
                'No projection or correction was applied.'
            )
        S = A_0 @ W @ A_0_inverse
        operators.append(validate_rotation_matrix(S, tolerance=tolerance, name='S'))
    return np.asarray(operators)


def rotation_matrix_residuals(matrix: np.ndarray) -> tuple[float, float]:
    """Return orthogonality and proper determinant residuals for a matrix.

    The first result is ``max(abs(R^T R - I))`` and the second is
    ``abs(det(R) - 1)``.  This function measures the supplied matrix without
    modifying it.
    """

    matrix = np.asarray(matrix, dtype=float)
    if matrix.size != 9:
        raise ValueError(f'A rotation matrix must have 9 values, not {matrix.size}')
    matrix = matrix.reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError('Rotation matrix contains a non-finite value')

    orthogonality = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    determinant_residual = float(abs(np.linalg.det(matrix) - 1.0))
    return orthogonality, determinant_residual


def validate_rotation_matrix(
    matrix: np.ndarray,
    *,
    tolerance: float = 1e-8,
    name: str = 'Matrix',
) -> np.ndarray:
    """Validate and return a proper rotation without altering its values."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.size != 9:
        raise ValueError(f'{name} must have 9 values, not {matrix.size}')
    matrix = matrix.reshape(3, 3)
    orthogonality, determinant_residual = rotation_matrix_residuals(matrix)
    if orthogonality > tolerance or determinant_residual > tolerance:
        raise ValueError(
            f'{name} is not a proper rotation matrix: '
            f'max|R^T R - I|={orthogonality:.3e}, '
            f'|det(R) - 1|={determinant_residual:.3e}, '
            f'tolerance={tolerance:.1e}. No projection or correction was applied.'
        )
    return matrix.copy()


def abc_to_orientation(
    abc_matrix: np.ndarray,
    *,
    rotation_tolerance: float = 1e-8,
    laue_class: str = 'm-3m',
    setting: str = 'standard',
) -> np.ndarray:
    """Calculate U = C^T A_0^-1 from a fixed-cell PolyLaue ABC matrix.

    PolyLaue stores the oriented cubic direct basis as rows, so

        C = ABC = a U^T,       U = C^T / a.

    The default m-3m path retains the original cubic calculation exactly.
    Other classes use the reference convention in fixed_reference_basis().
    The selected symmetry must preserve the stored cell metric.
    """

    abc = np.asarray(abc_matrix, dtype=float)
    if abc.size != 9:
        raise ValueError(f'An ABC matrix must have 9 values, not {abc.size}')

    abc = abc.reshape(3, 3)
    if not np.all(np.isfinite(abc)):
        raise ValueError('The ABC matrix contains a non-finite value')

    if laue_class != 'm-3m':
        A_0 = fixed_reference_basis(abc)
        proper_crystal_symmetry_operators(
            A_0, laue_class, setting, tolerance=rotation_tolerance
        )
        U = abc.T @ np.linalg.inv(A_0)
        return validate_rotation_matrix(
            U, tolerance=rotation_tolerance, name='U = C^T A_0^-1'
        )

    # Validate the setting even in the unchanged cubic path.
    _fractional_symmetry_operators(laue_class, setting)
    # The first stored direct vector has the fixed cubic length a.  The
    # subsequent proper-rotation validation independently checks that all
    # three vectors have that same length and are mutually perpendicular.
    lattice_parameter = float(np.linalg.norm(abc[0]))
    if not np.isfinite(lattice_parameter) or lattice_parameter <= 0:
        raise ValueError('The cubic ABC matrix has a non-positive lattice parameter')

    # This is the fixed-cell crystallographic relation itself
    orientation = abc.T / lattice_parameter
    return validate_rotation_matrix(
        orientation,
        tolerance=rotation_tolerance,
        name='U = ABC^T / a',
    )


def rotation_angle_deg(matrix: np.ndarray) -> float:
    """Return the principal rotation angle in degrees, in [0, 180]."""

    rotation = validate_rotation_matrix(matrix)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _cross_product_matrix(vector: np.ndarray) -> np.ndarray:
    """Return ``[v]_x``, defined by ``[v]_x a = v x a``."""

    x, y, z = np.asarray(vector).reshape(3)
    return np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ]
    )


def axis_angle_rotation_matrix(
    axis: np.ndarray | tuple[int, int, int],
    angle_deg: float,
) -> np.ndarray:
    """Construct a rotation matrix with Rodrigues' matrix equation.

    With unit axis ``n`` and ``K = [n]_x``, the literature form is

        R = cos(theta) I + (1 - cos(theta)) n n^T + sin(theta) K.
    """

    axis = np.asarray(axis, dtype=float).reshape(3)
    magnitude = np.linalg.norm(axis)
    if not np.isfinite(magnitude) or magnitude < 1e-12:
        raise ValueError('A rotation axis must be a finite non-zero vector')
    axis /= magnitude

    angle = math.radians(float(angle_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * np.eye(3)
        + (1.0 - cosine) * np.outer(axis, axis)
        + sine * _cross_product_matrix(axis)
    )


def cubic_csl_rotation_matrix(
    axis_uvw: tuple[int, int, int],
    x: int,
    y: int,
) -> tuple[np.ndarray, int, float]:
    """Return a cubic CSL rotation matrix, its Sigma, and its angle.

    This is Ranganathan's cubic axis--angle generating relation written
    directly as an integer numerator matrix.  Let

        d = [u, v, w],      p = y d,
        D = x^2 + p^T p,

        R = ((x^2 - p^T p) I + 2 p p^T + 2 x [p]_x) / D.

    Grimmer, Bollmann and Warrington show that the reduced common
    denominator of this rational cubic rotation matrix is ``Sigma``.  We
    therefore calculate Sigma by cancelling the integer common divisor of
    ``D`` and every element of the numerator matrix; it is not inferred from
    a lookup table or a different rotation representation.
    """

    axis = np.asarray(axis_uvw)
    if axis.shape != (3,) or not np.all(np.equal(axis, np.round(axis))):
        raise ValueError('axis_uvw must contain three integer indices')
    axis = axis.astype(int)

    x = int(x)
    y = int(y)
    if x < 0 or y < 0:
        raise ValueError('Ranganathan parameters x and y must be non-negative')
    if x == 0 and y == 0:
        raise ValueError('Ranganathan parameters x and y cannot both be zero')
    if y != 0 and not np.any(axis):
        raise ValueError('A non-zero y requires a non-zero rotation axis')

    p = y * axis
    p_squared = int(p @ p)
    denominator = x * x + p_squared
    numerator = (
        (x * x - p_squared) * np.eye(3, dtype=int)
        + 2 * np.outer(p, p)
        + 2 * x * _cross_product_matrix(p)
    ).astype(int)

    common_divisor = denominator
    for value in numerator.ravel():
        common_divisor = math.gcd(common_divisor, abs(int(value)))
    sigma = denominator // common_divisor

    rotation = numerator.astype(float) / denominator
    angle_deg = math.degrees(2.0 * math.atan2(math.sqrt(p_squared), x))
    return rotation, sigma, angle_deg


def rotation_axis(matrix: np.ndarray) -> np.ndarray:
    """Return the unit matrix-rotation axis; return NaNs for the identity.

    The axis is obtained from the defining matrix equation ``R n = n``.
    This eigenvector form remains stable for rotations close to 180 degrees,
    where division by ``sin(theta)`` in the skew-matrix formula is unstable.
    """

    rotation = validate_rotation_matrix(matrix)
    if (
        np.max(np.abs(rotation - np.eye(3))) < 1e-12
        or rotation_angle_deg(rotation) < 1e-10
    ):
        return np.full(3, np.nan)

    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    axis = np.real_if_close(eigenvectors[:, index], tol=1000).real
    magnitude = np.linalg.norm(axis)
    if magnitude < 1e-12:
        raise np.linalg.LinAlgError('Failed to extract a rotation-matrix axis')
    axis /= magnitude

    first_nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
    if first_nonzero.size and axis[first_nonzero[0]] < 0:
        axis *= -1
    return axis


def _canonical_matrix_key(matrix: np.ndarray) -> tuple[float, ...]:
    """Return a deterministic, rounded key using matrix elements only."""

    return tuple(np.round(validate_rotation_matrix(matrix).ravel(), 12))


def reduce_misorientation(
    matrix: np.ndarray,
    symmetries_1: np.ndarray,
    symmetries_2: np.ndarray | None = None,
) -> Disorientation:
    """Find the minimum-angle representative M_ij = S_i^T M S_j.

    Each S is a proper rotation in the Cartesian frame of its own crystal.
    If symmetries_2 is omitted, both grains use symmetries_1. Tied
    minimum-angle representatives are resolved deterministically; the angle
    is independent of that tie-breaking choice.

    For identical symmetry groups, cyclic invariance of the trace and group
    closure also permit a one-sided search for the minimum angle:

        tr(S_i^T M S_j) = tr(M S_j S_i^T),  S_j S_i^T in G.

    The explicit double-sided form is retained here because it represents the
    independent symmetry relabelling of both grains and returns the separate
    operators needed to express the selected axis in both original crystal
    frames.
    """

    matrix = validate_rotation_matrix(matrix)
    symmetries_1 = np.asarray(symmetries_1, dtype=float)
    symmetries_2 = symmetries_1 if symmetries_2 is None else np.asarray(
        symmetries_2, dtype=float
    )
    for operators in (symmetries_1, symmetries_2):
        if operators.ndim != 3 or operators.shape[1:] != (3, 3) or not len(operators):
            raise ValueError('Symmetries must be a non-empty array of 3 x 3 rotations')
        for S in operators:
            validate_rotation_matrix(S, name='Symmetry operator')

    # variants[i, j] = S_i.T @ matrix @ S_j
    variants = np.einsum(
        'aij,jk,bkl->abil',
        symmetries_1.transpose(0, 2, 1),
        matrix,
        symmetries_2,
    )
    traces = np.trace(variants, axis1=2, axis2=3)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    # A numerically identical pair has no unique axis. This addresses only
    # roundoff in the basis transformations, not experimental uncertainty.
    identities = np.max(np.abs(variants - np.eye(3)), axis=(2, 3)) < 1e-12
    angles[identities] = 0.0
    minimum = float(np.min(angles))

    tied_indices = np.argwhere(np.abs(angles - minimum) <= 1e-10)
    choices = []
    for i, j in tied_indices:
        candidate = variants[i, j]
        choices.append((_canonical_matrix_key(candidate), int(i), int(j)))

    _, i, j = min(choices, key=lambda item: item[0])
    return Disorientation(
        rotation=variants[i, j].copy(),
        angle_deg=math.degrees(minimum),
        symmetry_1=symmetries_1[i].copy(),
        symmetry_2=symmetries_2[j].copy(),
    )


def reduce_cubic_misorientation(matrix: np.ndarray) -> CubicDisorientation:
    """Retain the cubic API: test all 24 x 24 proper cubic symmetries."""

    return reduce_misorientation(matrix, proper_cubic_symmetry_operators())


def _reduce_unoriented_cubic_rotation(matrix: np.ndarray) -> CubicDisorientation:
    """Reduce a cubic rotation under symmetry and exchange of the grains.

    The double cubic-symmetry orbit is ``S_i^T R S_j``.  An unoriented grain
    pair additionally identifies ``R`` with ``R^-1 = R^T``, because exchanging
    grains 1 and 2 reverses the relative rotation.  Choosing the smaller
    matrix key from the reduced ``R`` and ``R^T`` orbits gives one
    deterministic representative of the complete unoriented CSL class.
    """

    matrix = validate_rotation_matrix(matrix)
    alternatives = (
        reduce_cubic_misorientation(matrix),
        reduce_cubic_misorientation(matrix.T),
    )
    return min(
        alternatives,
        key=lambda reduced: _canonical_matrix_key(reduced.rotation),
    )


@lru_cache(maxsize=None)
def _primitive_directions(max_index: int) -> tuple[tuple[int, int, int], ...]:
    directions = set()
    for h, k, l in itertools.product(range(-max_index, max_index + 1), repeat=3):
        if h == k == l == 0:
            continue
        divisor = math.gcd(math.gcd(abs(h), abs(k)), abs(l))
        if divisor != 1:
            continue

        direction = (h, k, l)
        first_nonzero = next(x for x in direction if x != 0)
        if first_nonzero < 0:
            direction = tuple(-x for x in direction)
        directions.add(direction)

    return tuple(sorted(directions, key=lambda x: (sum(v * v for v in x), x)))


def nearest_lattice_direction(
    vector: np.ndarray,
    max_index: int = 6,
    *,
    reference_basis: np.ndarray | None = None,
) -> tuple[tuple[int, int, int] | None, float]:
    """Return the nearest primitive integer direction and angular residual.

    The result is an approximation unless the residual is numerically zero.
    The direction is a line, so ``[u v w]`` and ``[-u -v -w]`` are treated as
    equivalent.

    With columns A_0 = [a_0 b_0 c_0], the angular comparison is
    cos(phi) = |v . (A_0 d)| / (|v| |A_0 d|), d = [u, v, w]^T.
    Omitting A_0 retains the original cubic integer-direction calculation.
    """

    vector = np.asarray(vector, dtype=float).reshape(3)
    magnitude = np.linalg.norm(vector)
    if not np.isfinite(magnitude) or magnitude < 1e-12:
        return None, math.nan
    vector = vector / magnitude
    A_0 = None if reference_basis is None else np.asarray(
        reference_basis, dtype=float
    ).reshape(3, 3)
    if A_0 is not None and (
        not np.all(np.isfinite(A_0)) or np.linalg.det(A_0) <= 0
    ):
        raise ValueError('The reference basis must be finite and right-handed')

    best_direction = None
    best_cosine = -1.0
    best_norm_squared = math.inf
    for direction in _primitive_directions(max_index):
        direction_array = np.asarray(direction, dtype=float)
        if A_0 is not None:
            direction_array = A_0 @ direction_array
        norm_squared = float(direction_array @ direction_array)
        cosine = abs(float(vector @ direction_array) / math.sqrt(norm_squared))
        if cosine > best_cosine + 1e-14 or (
            abs(cosine - best_cosine) <= 1e-14 and norm_squared < best_norm_squared
        ):
            best_direction = direction
            best_cosine = cosine
            best_norm_squared = norm_squared

    error = math.degrees(math.acos(np.clip(best_cosine, -1.0, 1.0)))
    return best_direction, error


def _canonical_cubic_direction_family(
    direction: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    """Put a cubic direction into the conventional ``<u v w>`` family."""

    if direction is None:
        return None
    return tuple(sorted((abs(value) for value in direction), reverse=True))


def analyze_orientation_relationship(
    abc_matrix_1: np.ndarray,
    abc_matrix_2: np.ndarray,
    direction_max_index: int = 6,
    *,
    laue_class: str = 'm-3m',
    setting: str = 'standard',
) -> OrientationRelationship:
    """Calculate raw and symmetry-reduced relationships of two fixed cells.

    Both grains are assumed to be the same phase, with the selected Laue
    class and cell setting. Phase identity is not inferred or verified.
    """

    orientation_1 = abc_to_orientation(
        abc_matrix_1, laue_class=laue_class, setting=setting
    )
    orientation_2 = abc_to_orientation(
        abc_matrix_2, laue_class=laue_class, setting=setting
    )

    # M maps grain-2 crystal components into grain-1 crystal components for
    # the same laboratory vector.
    raw = validate_rotation_matrix(
        orientation_1.T @ orientation_2,
        name='M = U1^T U2',
    )
    raw_angle = rotation_angle_deg(raw)
    A_01 = A_02 = None
    if laue_class == 'm-3m':
        reduced = reduce_cubic_misorientation(raw)
    else:
        A_01 = orientation_1.T @ np.asarray(abc_matrix_1).reshape(3, 3).T
        A_02 = orientation_2.T @ np.asarray(abc_matrix_2).reshape(3, 3).T
        reduced = reduce_misorientation(
            raw,
            proper_crystal_symmetry_operators(A_01, laue_class, setting),
            proper_crystal_symmetry_operators(A_02, laue_class, setting),
        )
    reduced_axis = rotation_axis(reduced.rotation)

    if np.any(np.isnan(reduced_axis)):
        axis_lab = np.full(3, np.nan)
        axis_crystal_1 = np.full(3, np.nan)
        axis_crystal_2 = np.full(3, np.nan)
    else:
        axis_crystal_1 = reduced.symmetry_1 @ reduced_axis
        axis_crystal_2 = reduced.symmetry_2 @ reduced_axis
        axis_lab = orientation_1 @ axis_crystal_1

        # An axis is an unoriented line.  Flip all three representations
        # together to make the displayed result deterministic.
        first_nonzero = np.flatnonzero(np.abs(axis_lab) > 1e-12)
        if first_nonzero.size and axis_lab[first_nonzero[0]] < 0:
            axis_lab *= -1
            axis_crystal_1 *= -1
            axis_crystal_2 *= -1

    uvw_1, uvw_error_1 = nearest_lattice_direction(
        axis_crystal_1, max_index=direction_max_index, reference_basis=A_01
    )
    uvw_2, uvw_error_2 = nearest_lattice_direction(
        axis_crystal_2, max_index=direction_max_index, reference_basis=A_02
    )

    return OrientationRelationship(
        orientation_1=orientation_1,
        orientation_2=orientation_2,
        raw_misorientation=raw,
        raw_angle_deg=raw_angle,
        disorientation=reduced.rotation,
        angle_deg=reduced.angle_deg,
        axis_lab=axis_lab,
        axis_crystal_1=axis_crystal_1,
        axis_crystal_2=axis_crystal_2,
        axis_uvw_1=uvw_1,
        axis_uvw_2=uvw_2,
        axis_uvw_error_deg_1=uvw_error_1,
        axis_uvw_error_deg_2=uvw_error_2,
        symmetry_1=reduced.symmetry_1,
        symmetry_2=reduced.symmetry_2,
        laue_class=laue_class,
        setting=setting,
    )


def _variant_suffix(index: int, count: int) -> str:
    if count == 1:
        return ''
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    return f'-{index + 1}'


@lru_cache(maxsize=None)
def generate_cubic_csls(max_sigma: int = 49) -> tuple[CubicCsl, ...]:
    """Generate unoriented, symmetry-distinct exact cubic CSL rotations.

    This enumerates Ranganathan's matrix parameters: primitive cubic axes
    ``[u v w]`` and coprime integer pairs ``(x, y)``.  Cubic symmetry lets us
    restrict the axis to ``u >= v >= w >= 0``.  The integer numerator and
    denominator of every candidate rotation are passed to
    :func:`cubic_csl_rotation_matrix`, which obtains Sigma from the reduced
    matrix denominator.

    For primitive parameters, at most a factor of four cancels from the
    matrix denominator.  Consequently, a complete search through
    ``Sigma <= max_sigma`` requires only candidates whose unreduced
    denominator is at most ``4 * max_sigma``.

    The catalogue also identifies ``R`` and ``R^T``.  These rotations are
    related by exchanging the two grains and therefore describe the same
    unoriented CSL relationship.
    """

    max_sigma = int(max_sigma)
    if max_sigma < 1:
        raise ValueError('max_sigma must be at least 1')

    denominator_limit = 4 * max_sigma
    parameter_limit = math.isqrt(denominator_limit)
    unique: dict[tuple[object, ...], CubicDisorientation] = {}

    identity, sigma, _ = cubic_csl_rotation_matrix((0, 0, 1), x=1, y=0)
    identity_reduced = _reduce_unoriented_cubic_rotation(identity)
    identity_key = (sigma, *_canonical_matrix_key(identity_reduced.rotation))
    unique[identity_key] = identity_reduced

    for u in range(parameter_limit + 1):
        for v in range(u + 1):
            for w in range(v + 1):
                axis = (u, v, w)
                if axis == (0, 0, 0):
                    continue
                if math.gcd(math.gcd(u, v), w) != 1:
                    continue

                axis_squared = u * u + v * v + w * w
                for x in range(parameter_limit + 1):
                    remaining = denominator_limit - x * x
                    if remaining < axis_squared:
                        continue
                    y_limit = math.isqrt(remaining // axis_squared)

                    for y in range(1, y_limit + 1):
                        if math.gcd(x, y) != 1:
                            continue

                        exact_rotation, sigma, _ = cubic_csl_rotation_matrix(
                            axis,
                            x,
                            y,
                        )
                        if sigma > max_sigma:
                            continue

                        reduced = _reduce_unoriented_cubic_rotation(exact_rotation)
                        key = (
                            sigma,
                            *_canonical_matrix_key(reduced.rotation),
                        )
                        unique.setdefault(key, reduced)

    grouped: dict[int, list[CubicDisorientation]] = {}
    for key, value in unique.items():
        grouped.setdefault(int(key[0]), []).append(value)

    results = []
    for sigma in sorted(grouped):
        group = grouped[sigma]
        group.sort(
            key=lambda reduced: (
                reduced.angle_deg,
                _canonical_matrix_key(reduced.rotation),
            )
        )
        for index, reduced in enumerate(group):
            axis = rotation_axis(reduced.rotation)
            uvw, _ = nearest_lattice_direction(axis, max_index=12)
            uvw = _canonical_cubic_direction_family(uvw)
            rotation = reduced.rotation.copy()
            rotation.setflags(write=False)
            axis = axis.copy()
            axis.setflags(write=False)
            results.append(
                CubicCsl(
                    sigma=sigma,
                    variant=_variant_suffix(index, len(group)),
                    rotation=rotation,
                    angle_deg=reduced.angle_deg,
                    axis_crystal=axis,
                    axis_uvw=uvw,
                )
            )

    return tuple(results)


def rotation_distance_deg(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    """Geodesic angular distance between two proper rotations."""

    matrix_1 = validate_rotation_matrix(matrix_1, name='First rotation')
    matrix_2 = validate_rotation_matrix(matrix_2, name='Second rotation')
    return rotation_angle_deg(matrix_1 @ matrix_2.T)


def _distance_to_csl(misorientation: np.ndarray, csl_rotation: np.ndarray) -> float:
    """Minimum full-rotation distance to a cubic-equivalent exact CSL."""

    misorientation = validate_rotation_matrix(
        misorientation,
        name='Measured misorientation',
    )
    csl_rotation = validate_rotation_matrix(
        csl_rotation,
        name='Exact CSL rotation',
    )
    symmetries = proper_cubic_symmetry_operators()
    minimum = math.inf

    # Transposition includes exchange of the two grains.  It has no effect on
    # Sigma but is part of an unoriented grain-boundary classification.
    for exact in (csl_rotation, csl_rotation.T):
        variants = np.einsum(
            'aij,jk,bkl->abil',
            symmetries.transpose(0, 2, 1),
            exact,
            symmetries,
        )
        differences = np.matmul(
            misorientation,
            np.swapaxes(variants, -1, -2),
        )
        traces = np.trace(differences, axis1=2, axis2=3)
        angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
        minimum = min(minimum, float(np.min(angles)))

    return math.degrees(minimum)


def match_cubic_csls(
    misorientation: np.ndarray,
    max_sigma: int = 49,
    brandon_constant_deg: float = 15.0,
) -> tuple[CubicCslMatch, ...]:
    """Compare a measured rotation with exact cubic CSL misorientations.

    Results are sorted by the full rotation-space deviation.  Matching is
    not performed on angle alone.  ``brandon_constant_deg`` is exposed so
    the empirical acceptance convention is explicit rather than hidden.
    """

    if brandon_constant_deg <= 0:
        raise ValueError('brandon_constant_deg must be positive')

    matches = []
    for csl in generate_cubic_csls(max_sigma):
        deviation = _distance_to_csl(misorientation, csl.rotation)
        limit = float(brandon_constant_deg) / math.sqrt(csl.sigma)
        matches.append(
            CubicCslMatch(
                csl=csl,
                deviation_deg=deviation,
                brandon_limit_deg=limit,
            )
        )

    matches.sort(
        key=lambda match: (
            match.deviation_deg,
            match.csl.sigma,
            match.csl.variant,
        )
    )
    return tuple(matches)
