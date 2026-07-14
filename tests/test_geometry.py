"""Tests for named geometry and coordinate-frame value objects."""

from dataclasses import FrozenInstanceError

import numpy as np
import numpy.testing as npt
import pytest

from geodef.geometry import LocalFrame, PlanarGeometry, TriGeometry


class TestLocalFrame:
    """LocalFrame validation, conversion, and compatibility."""

    def test_records_projection_and_is_immutable(self) -> None:
        frame = LocalFrame(1.0, 100.0, origin_alt=12.0)

        assert frame.projection == "wgs84-enu"
        with pytest.raises(FrozenInstanceError):
            frame.origin_lat = 2.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"origin_lat": 91.0, "origin_lon": 0.0}, "origin_lat"),
            ({"origin_lat": 0.0, "origin_lon": np.nan}, "origin_lon"),
            (
                {
                    "origin_lat": 0.0,
                    "origin_lon": 0.0,
                    "projection": "utm",
                },
                "projection",
            ),
        ],
    )
    def test_invalid_definition_raises(
        self, kwargs: dict[str, float | str], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            LocalFrame(**kwargs)  # type: ignore[arg-type]

    def test_geographic_enu_round_trip(self) -> None:
        frame = LocalFrame(1.0, 100.0, origin_alt=25.0)
        lon = np.array([100.0, 100.02])
        lat = np.array([1.0, 1.01])
        alt = np.array([25.0, -1500.0])

        enu = frame.to_enu(lon=lon, lat=lat, alt=alt)
        geographic = frame.to_geographic(east=enu[:, 0], north=enu[:, 1], up=enu[:, 2])

        assert enu.shape == (2, 3)
        npt.assert_allclose(geographic[:, 0], lon, atol=1e-10)
        npt.assert_allclose(geographic[:, 1], lat, atol=1e-10)
        npt.assert_allclose(geographic[:, 2], alt, atol=1e-6)

    def test_explicit_transform_between_frames(self) -> None:
        source = LocalFrame(0.0, 100.0)
        target = LocalFrame(0.1, 100.2)
        coordinates = np.array([[0.0, 0.0, -1000.0], [500.0, 200.0, 0.0]])

        transformed = source.transform_enu(coordinates, target=target)
        geographic = source.to_geographic(
            east=coordinates[:, 0],
            north=coordinates[:, 1],
            up=coordinates[:, 2],
        )
        expected = target.to_enu(
            lon=geographic[:, 0],
            lat=geographic[:, 1],
            alt=geographic[:, 2],
        )

        npt.assert_allclose(transformed, expected)

    def test_incompatible_frames_are_rejected(self) -> None:
        frame = LocalFrame(0.0, 100.0)
        other = LocalFrame(0.0, 101.0)

        assert frame.is_compatible(frame)
        assert not frame.is_compatible(other)
        with pytest.raises(ValueError, match="incompatible local frames"):
            frame.require_compatible(other)


class TestPlanarGeometry:
    """PlanarGeometry named and array views."""

    def test_named_views_and_theta(self) -> None:
        frame = LocalFrame(-2.0, 100.0)
        geometry = PlanarGeometry(
            center=(1200.0, -500.0),
            depth=15_000.0,
            strike=315.0,
            dip=25.0,
            length=80_000.0,
            width=40_000.0,
            frame=frame,
        )

        npt.assert_array_equal(
            geometry.theta,
            [1200.0, -500.0, 15_000.0, 315.0, 25.0, 80_000.0, 40_000.0],
        )
        npt.assert_array_equal(geometry.to_enu(), [1200.0, -500.0, -15_000.0])
        assert geometry.to_geographic().shape == (3,)
        assert not geometry.theta.flags.writeable

    def test_from_geographic_round_trip(self) -> None:
        frame = LocalFrame(-2.0, 100.0)
        geometry = PlanarGeometry.from_geographic(
            lon=100.1,
            lat=-1.9,
            depth=12_000.0,
            strike=30.0,
            dip=45.0,
            length=20_000.0,
            width=10_000.0,
            frame=frame,
        )

        lon, lat, depth = geometry.to_geographic()
        assert lon == pytest.approx(100.1)
        assert lat == pytest.approx(-1.9)
        assert depth == pytest.approx(12_000.0)

    def test_from_theta_and_to_frame_preserve_physical_center(self) -> None:
        source = LocalFrame(0.0, 100.0)
        target = LocalFrame(0.1, 100.2)
        geometry = PlanarGeometry.from_theta(
            [500.0, -300.0, 10_000.0, 5.0, 40.0, 20_000.0, 8_000.0],
            frame=source,
        )

        transformed = geometry.to_frame(target)

        npt.assert_allclose(
            transformed.to_geographic(), geometry.to_geographic(), atol=1e-6
        )
        assert transformed.frame == target
        assert transformed.depth == geometry.depth

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("center", (0.0,)),
            ("depth", -1.0),
            ("strike", 360.0),
            ("dip", 91.0),
            ("length", 0.0),
            ("width", np.inf),
        ],
    )
    def test_invalid_geometry_raises(self, field: str, value: object) -> None:
        kwargs: dict[str, object] = {
            "center": (0.0, 0.0),
            "depth": 10_000.0,
            "strike": 0.0,
            "dip": 30.0,
            "length": 20_000.0,
            "width": 10_000.0,
            "frame": LocalFrame(0.0, 100.0),
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=field):
            PlanarGeometry(**kwargs)  # type: ignore[arg-type]


class TestTriGeometry:
    """TriGeometry construction, derived values, and frame transforms."""

    @pytest.fixture
    def geometry(self) -> TriGeometry:
        vertices = np.array(
            [
                [[0.0, 0.0, -1000.0], [1000.0, 0.0, -1000.0], [0.0, 0.0, -2000.0]],
                [
                    [1000.0, 0.0, -1000.0],
                    [1000.0, 0.0, -2000.0],
                    [0.0, 0.0, -2000.0],
                ],
            ]
        )
        return TriGeometry(vertices_enu=vertices, frame=LocalFrame(0.0, 100.0))

    def test_named_views_and_derived_orientation(self, geometry: TriGeometry) -> None:
        assert geometry.n_triangles == 2
        assert geometry.to_enu().shape == (2, 3, 3)
        assert geometry.to_geographic().shape == (2, 3, 3)
        assert geometry.centers_enu.shape == (2, 3)
        assert geometry.centers_geographic.shape == (2, 3)
        npt.assert_allclose(geometry.dip, 90.0)
        assert np.all((geometry.strike >= 0.0) & (geometry.strike < 360.0))

    def test_owns_read_only_copy(self) -> None:
        vertices = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]])
        geometry = TriGeometry(vertices, LocalFrame(0.0, 0.0))
        vertices[0, 0, 0] = 999.0

        assert geometry.vertices_enu[0, 0, 0] == 0.0
        with pytest.raises(ValueError):
            geometry.vertices_enu[0, 0, 0] = 1.0

    def test_from_nodes_preserves_connectivity_order(self) -> None:
        nodes = np.array(
            [
                [0.0, 0.0, -1000.0],
                [1000.0, 0.0, -1000.0],
                [0.0, 1000.0, -2000.0],
                [1000.0, 1000.0, -2000.0],
            ]
        )
        triangles = np.array([[1, 3, 2], [0, 1, 2]])

        geometry = TriGeometry.from_nodes(
            nodes, triangles, frame=LocalFrame(0.0, 100.0)
        )

        npt.assert_array_equal(geometry.vertices_enu[0], nodes[triangles[0]])

    def test_from_geographic_and_to_frame_round_trip(self) -> None:
        lon = np.array([100.0, 100.01, 100.0])
        lat = np.array([0.0, 0.0, 0.01])
        depth = np.array([1000.0, 1000.0, 2000.0])
        triangles = np.array([[0, 1, 2]])
        source = LocalFrame(0.0, 100.0)

        geometry = TriGeometry.from_geographic(
            lon=lon,
            lat=lat,
            depth=depth,
            triangles=triangles,
            frame=source,
        )
        transformed = geometry.to_frame(LocalFrame(0.05, 100.05))

        npt.assert_allclose(
            transformed.to_geographic(), geometry.to_geographic(), atol=1e-6
        )

    def test_invalid_vertices_raise(self) -> None:
        frame = LocalFrame(0.0, 0.0)
        with pytest.raises(ValueError, match="vertices_enu"):
            TriGeometry(np.zeros((3, 3)), frame)
        with pytest.raises(ValueError, match="finite"):
            TriGeometry(np.full((1, 3, 3), np.nan), frame)
