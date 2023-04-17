import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x * y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a * c
    if den > 0:
        raise ValueError("coeffs do not represent an ellipse: b^2 - 4ac must" " be negative!")

    # The location of the ellipse centre.
    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2.0 * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def rotation_matrix(angle: float, axis: str):
    if axis == "x":
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
    elif axis == "y":
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
    elif axis == "z":
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )


def generate_data(
    c: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n: int = 100,
    phi: float = 2 * np.pi,
    noise_range: float = 0.1,
) -> np.ndarray:
    result = np.zeros(shape=(n, 3))

    for i in range(n):
        t = i / n * phi
        result[i] = c + u * np.cos(t) + v * np.sin(t)

    noise = np.random.uniform(-noise_range, noise_range, size=(n, 3))

    return result + noise


def normalize(v: np.ndarray) -> np.ndarray:
    """
    v != [0, 0, 0]
    """
    return v / (v**2).sum() ** 0.5


def get_normal_vector(data: np.ndarray) -> np.ndarray:
    # one might use svd transformation to do the same thing, it is more concise
    xi, yi, zi = data[:, 0], data[:, 1], data[:, 2]

    B = np.array(
        [
            [(xi * zi).sum()],
            [(yi * zi).sum()],
            [zi.sum()],
        ]
    )

    A = np.array(
        [
            [(xi * xi).sum(), (xi * yi).sum(), xi.sum()],
            [(xi * yi).sum(), (yi * yi).sum(), yi.sum()],
            [xi.sum(), yi.sum(), len(xi)],
        ]
    )

    res = np.linalg.inv(A) @ B
    return normalize(np.array([res[0] / res[2], res[1] / res[2], -1 / res[2]]))


def project_data_on_2d(data: np.ndarray) -> np.ndarray:
    """
    find approximating plane and its normal vector using least squares method
    """

    normal_vector = get_normal_vector(data)
    rot, _ = Rotation.align_vectors(normal_vector.T, np.array([[0, 0, 1]]))
    rotation = Rotation.as_matrix(rot)
    new_data = (data - data.mean(axis=0)) @ rotation

    return new_data[:, 0:2]


def fit_2d_ellipse(data: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """
    returns semi-major axis, eccentricity and centre coordinates
    """
    params = fit_ellipse(data[:, 0], data[:, 1])
    x0, y0, ap, bp, e, phi = cart_to_pol(params)
    return ap, e, (x0, y0)


def fit_3d_ellipse(data: np.ndarray) -> tuple[float, float]:
    """
    takes a set of points in array (N, 3) of 3d points and approximates them with an ellipse by
    projecting them onto a plane and then approximating with 2d ellipse.

    Returns: semi-major axis, eccentricity
    """
    data_2d = project_data_on_2d(data)
    sma, e, _ = fit_2d_ellipse(data_2d)
    return sma, e


if __name__ == "__main__":
    ##### generate data

    centre = np.array([1, 2, 1])
    u = np.array([2, -np.sqrt(2), 0]) * 2
    v = np.array([1, np.sqrt(2), 2])
    print(f"u: {(u ** 2).sum() ** 0.5}")
    print(f"v: {(v ** 2).sum() ** 0.5}")

    assert np.abs(np.dot(u, v)) < 1e-10

    initial_rotation = rotation_matrix(np.pi / 4 * 1.5, "x")
    data = generate_data(centre, u, v, phi=0.4 * np.pi, n=10000)

    eccentricity = np.sqrt(1 - (v**2).sum() / (u**2).sum())
    semimajor_axis = np.sqrt((u**2).sum())

    ##### computation

    normal_vector = get_normal_vector(data)
    data_2d = project_data_on_2d(data)
    sma, e, approx_centre = fit_2d_ellipse(data_2d)

    print(f"a: {sma}")
    print(f"e: {e}")

    ##### plot data and its semi- major and minor axes

    fig = plt.figure(figsize=plt.figaspect(2))
    fig.suptitle(f"e: {eccentricity:.02f}; a: {semimajor_axis:.02f}")

    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_zlim(-4, 4)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.grid(True)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)

    ax1.plot(
        [centre[0] + u[0], centre[0]],
        [centre[1] + u[1], centre[1]],
        [centre[2] + u[2], centre[2]],
        color="g",
        linestyle="dashed",
    )
    ax1.plot(
        [centre[0] + v[0], centre[0]],
        [centre[1] + v[1], centre[1]],
        [centre[2] + v[2], centre[2]],
        color="g",
        linestyle="dashed",
    )
    ax1.plot(data[:, 0], data[:, 1], data[:, 2], marker=",", linestyle="none", color="g")
    ax1.plot(
        [centre[0], centre[0] + normal_vector[0]],
        [centre[1], centre[1] + normal_vector[1]],
        [centre[2], centre[2] + normal_vector[2]],
        color="g",
    )

    ax2.plot(data_2d[:, 0], data_2d[:, 1], linestyle="none", marker=",")
    ax2.plot(*approx_centre, "ro")
    ax2.plot(*data_2d.mean(axis=0), "bo")

    plt.show()
