"""Verify the supplied robot archives and extract immutable STL inputs.

The 3MF is a slicer project: its transforms arrange parts on print plates rather
than assembling the arm.  The source ZIP contains the same printable parts by
name.  This script verifies both source hashes, validates every STL against the
checked-in manifest, and optionally extracts them to the ignored generated/
tree for MuJoCo.  It never rewrites the user's source archives.
"""

from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import zipfile

import trimesh


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "model_manifest.json"
GENERATED_MESH_DIR = HERE / "generated" / "meshes"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path=MANIFEST_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported simulation manifest format")
    return payload


def _safe_member(name):
    path = Path(name)
    return (not path.is_absolute() and ".." not in path.parts
            and path.name == name and path.suffix.lower() == ".stl")


def inspect_sources(manifest=None, tolerance_mm=0.06):
    manifest = manifest or load_manifest()
    source = manifest["source"]
    archive = HERE / source["archive"]
    model_3mf = HERE / source["model_3mf"]
    if sha256(archive) != source["archive_sha256"]:
        raise RuntimeError(f"source ZIP hash mismatch: {archive}")
    if sha256(model_3mf) != source["model_3mf_sha256"]:
        raise RuntimeError(f"source 3MF hash mismatch: {model_3mf}")

    expected = {item["name"]: item for item in manifest["meshes"]}
    observed = {}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if any(not _safe_member(name) for name in names):
            raise RuntimeError("source ZIP contains an unsafe/non-STL member")
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise RuntimeError(f"mesh set mismatch; missing={missing}, extra={extra}")
        for name in names:
            mesh = trimesh.load(io.BytesIO(bundle.read(name)), file_type="stl")
            extents = [float(value) for value in mesh.extents]
            reference = expected[name]
            if len(mesh.faces) != reference["faces"]:
                raise RuntimeError(f"{name}: face count changed")
            if not mesh.is_watertight:
                raise RuntimeError(f"{name}: mesh is no longer watertight")
            if any(not math.isfinite(value) or value <= 0 for value in extents):
                raise RuntimeError(f"{name}: invalid extents {extents}")
            if any(abs(actual - wanted) > tolerance_mm
                   for actual, wanted in zip(extents, reference["extents_mm"])):
                raise RuntimeError(
                    f"{name}: extent mismatch {extents} vs {reference['extents_mm']}")
            observed[name] = {
                "faces": len(mesh.faces),
                "extents_mm": extents,
                "watertight": bool(mesh.is_watertight),
            }
    return observed


def extract_sources(manifest=None, destination=GENERATED_MESH_DIR):
    manifest = manifest or load_manifest()
    inspect_sources(manifest)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = HERE / manifest["source"]["archive"]
    expected = {item["name"] for item in manifest["meshes"]}
    with zipfile.ZipFile(archive) as bundle:
        for name in sorted(expected):
            data = bundle.read(name)
            target = destination / name
            if target.exists() and target.read_bytes() == data:
                continue
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true",
                        help="also populate ignored generated/meshes")
    parser.add_argument("--destination", type=Path,
                        default=GENERATED_MESH_DIR)
    args = parser.parse_args()
    manifest = load_manifest()
    observed = inspect_sources(manifest)
    print(
        f"ASSETS_OK meshes={len(observed)} "
        f"zip={manifest['source']['archive_sha256'][:12]} "
        f"3mf={manifest['source']['model_3mf_sha256'][:12]}")
    if args.extract:
        destination = extract_sources(manifest, args.destination)
        print(f"EXTRACTED {len(observed)} meshes -> {destination}")
    return True


if __name__ == "__main__":
    main()
