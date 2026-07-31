"""Minimal VTK ImageData (.vti) writer.

MAR sits on a uniform grid -- 20 km spacing, constant origin -- so vtkImageData
is the right container: it stores origin, spacing and dimensions rather than a
coordinate per point, which is both smaller and what ParaView handles fastest.

Hand-written rather than pulling in vtk or pyevtk. The format is a short XML
header followed by raw little-endian arrays, each prefixed by its byte count,
and writing it directly avoids a heavyweight runtime dependency for what is
about sixty lines. Output is validated against the real vtk reader in the
project's test invocation, not assumed to be correct.

Point ordering in VTK is x fastest, then y, then z. A numpy array shaped
(nz, ny, nx) ravelled in C order already satisfies that, so callers pass arrays
in that shape and no transpose happens here.
"""

import struct
from pathlib import Path

import numpy as np

_VTK_TYPE = {
    np.dtype(np.float32): "Float32",
    np.dtype(np.float64): "Float64",
    np.dtype(np.int32): "Int32",
    np.dtype(np.uint8): "UInt8",
}


def write_image_data(
    path: Path,
    fields: dict[str, np.ndarray],
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> Path:
    """Write point-centred `fields` to `path` as a binary .vti.

    Every array must share the same (nz, ny, nx) shape. Values are written as
    float32; NaN is preserved and ParaView treats it as blank, which is how the
    non-ice cells are excluded from colour scales.
    """
    if not fields:
        raise ValueError("no fields to write")

    shapes = {array.shape for array in fields.values()}
    if len(shapes) != 1:
        raise ValueError(f"fields have mismatched shapes: {shapes}")
    n_z, n_y, n_x = shapes.pop()

    payloads = []
    offset = 0
    declarations = []
    for name, array in fields.items():
        flat = np.ascontiguousarray(array, dtype=np.float32).ravel(order="C")
        declarations.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="1" format="appended" offset="{offset}"/>'
        )
        payloads.append(flat)
        # Each appended block is a UInt64 byte count followed by the raw bytes.
        offset += 8 + flat.nbytes

    extent = f"0 {n_x - 1} 0 {n_y - 1} 0 {n_z - 1}"
    header = "\n".join(
        [
            '<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64">',
            f'  <ImageData WholeExtent="{extent}" '
            f'Origin="{origin[0]} {origin[1]} {origin[2]}" '
            f'Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">',
            f'    <Piece Extent="{extent}">',
            f'      <PointData Scalars="{next(iter(fields))}">',
            *declarations,
            "      </PointData>",
            "    </Piece>",
            "  </ImageData>",
            '  <AppendedData encoding="raw">',
            "   _",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        for flat in payloads:
            handle.write(struct.pack("<Q", flat.nbytes))
            handle.write(flat.tobytes(order="C"))
        handle.write(b"\n  </AppendedData>\n</VTKFile>\n")
    return path


def write_time_series_index(path: Path, entries: list[tuple[float, str]]) -> Path:
    """Write a .pvd collection so ParaView sees one dataset with a time axis.

    Without this ParaView loads the files as an unordered group and the
    animation controls carry no real time values.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
        "  <Collection>",
        *[
            f'    <DataSet timestep="{time}" group="" part="0" file="{name}"/>'
            for time, name in entries
        ],
        "  </Collection>",
        "</VTKFile>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
