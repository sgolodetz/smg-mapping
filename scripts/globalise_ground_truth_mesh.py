import numpy as np
import open3d as o3d
import re

from typing import Dict

from smg.open3d import VisualisationUtil
from smg.utility import GeometryUtil


def load_fiducials(filename: str) -> Dict[str, np.ndarray]:
    """
    Load named fiducials from a file.

    :param filename:    The name of the file containing the fiducials.
    :return:            A dictionary mapping fiducial names to positions in 3D space.
    """
    fiducials: Dict[str, np.ndarray] = {}

    vec_spec = r"\(\s*(.*?)\s+(.*?)\s+(.*?)\s*\)"
    line_spec = r".*?\s+(.*?)\s+" + vec_spec + r"\s+?" + vec_spec + r"\s+?" + vec_spec + r"\s+?" + vec_spec + ".*"
    prog = re.compile(line_spec)

    def to_vec(x: int, y: int, z: int) -> np.ndarray:
        return np.array([float(m.group(x)), float(m.group(y)), float(m.group(z))])

    with open(filename, "r") as f:
        for line in f:
            m = prog.match(line)
            name = m.group(1)
            pos = to_vec(2, 3, 4)
            fiducials[name] = pos

    return fiducials


def main() -> None:
    fiducials: Dict[str, np.ndarray] = load_fiducials(
        "C:/spaint/build/bin/apps/spaintgui/meshes/fiducials-20210124T214842.txt"
    )
    print(fiducials)

    p: np.ndarray = np.column_stack([
        fiducials["0_0"],
        fiducials["0_1"],
        fiducials["0_2"],
        fiducials["0_3"]
    ])

    height: float = 1.5  # 1.5m (the height of the centre of the printed marker)
    offset: float = 0.0705  # 7.05cm (half the width of the printed marker)

    q: np.ndarray = np.array([
        [-offset, -(height + offset), 0],
        [offset, -(height + offset), 0],
        [offset, -(height - offset), 0],
        [-offset, -(height - offset), 0]
    ]).transpose()

    print(p.shape)
    print(p)
    print(q)

    transform: np.ndarray = GeometryUtil.estimate_rigid_transform(p, q)
    print(transform)

    # noinspection PyUnresolvedReferences
    mesh: o3d.geometry.TrianglMesh = o3d.io.read_triangle_mesh(
        "C:/spaint/build/bin/apps/spaintgui/meshes/spaint-20210124T214842_World.ply"
    )
    mesh = mesh.transform(transform)

    VisualisationUtil.visualise_geometry(mesh)


if __name__ == "__main__":
    main()
