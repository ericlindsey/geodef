"""Tests for coordinate frames and geometry-array functions."""

from dataclasses import FrozenInstanceError

import numpy as np
import numpy.testing as npt
import pytest

from geodef.geometry import (
    as_planar_vector,
    planar_parameter_dict,
    triangle_strike_dip,
    vertices_from_nodes,
)
from geodef.transforms import LocalFrame


class TestLocalFrame:
    """LocalFrame validation, conversion, and compatibility."""

    def test_is_immutable(self) -> None:
        frame = LocalFrame(lon=100.0, lat=1.0, alt=12.0)

        with pytest.raises(FrozenInstanceError):
            frame.lat = 2.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lat": 91.0, "lon": 0.0}, "lat"),
            ({"lat": 0.0, "lon": np.nan}, "lon"),
        ],
    )
    def test_invalid_definition_raises(
        self, kwargs: dict[str, float | str], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            LocalFrame(**kwargs)  # type: ignore[arg-type]

    def test_geographic_enu_round_trip(self) -> None:
        frame = LocalFrame(lon=100.0, lat=1.0, alt=25.0)
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
        source = LocalFrame(lon=100.0, lat=0.0)
        target = LocalFrame(lon=100.2, lat=0.1)
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
        frame = LocalFrame(lon=100.0, lat=0.0)
        other = LocalFrame(lon=101.0, lat=0.0)

        assert frame.is_compatible(frame)
        assert not frame.is_compatible(other)
        with pytest.raises(ValueError, match="incompatible local frames"):
            frame.require_compatible(other)


class TestPlanarFunctions:
    """Planar parameter mappings and expert vectors round-trip."""

    def test_mapping_to_vector_and_back(self) -> None:
        parameters = {
            "depth": 15_000.0,
            "e0": 1200.0,
            "n0": -500.0,
            "strike": 315.0,
            "dip": 25.0,
            "length": 80_000.0,
            "width": 40_000.0,
        }

        vector = as_planar_vector(parameters)

        npt.assert_array_equal(
            vector,
            [1200.0, -500.0, 15_000.0, 315.0, 25.0, 80_000.0, 40_000.0],
        )
        assert planar_parameter_dict(vector) == parameters

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("depth", -1.0),
            ("strike", 360.0),
            ("dip", 91.0),
            ("length", 0.0),
            ("width", np.inf),
        ],
    )
    def test_invalid_parameter_raises(self, field: str, value: float) -> None:
        parameters = {
            "e0": 0.0,
            "n0": 0.0,
            "depth": 10_000.0,
            "strike": 0.0,
            "dip": 30.0,
            "length": 20_000.0,
            "width": 10_000.0,
        }
        parameters[field] = value

        with pytest.raises(ValueError, match=field):
            as_planar_vector(parameters)


class TestTriangleFunctions:
    """Triangle expansion and orientation functions."""

    def test_orientation(self) -> None:
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

        strike, dip = triangle_strike_dip(vertices)

        npt.assert_allclose(dip, 90.0)
        assert np.all((strike >= 0.0) & (strike < 360.0))

    def test_vertices_from_nodes_preserves_connectivity_order(self) -> None:
        nodes = np.array(
            [
                [0.0, 0.0, -1000.0],
                [1000.0, 0.0, -1000.0],
                [0.0, 1000.0, -2000.0],
                [1000.0, 1000.0, -2000.0],
            ]
        )
        triangles = np.array([[1, 3, 2], [0, 1, 2]])

        vertices = vertices_from_nodes(nodes, triangles)

        npt.assert_array_equal(vertices[0], nodes[triangles[0]])

    def test_vertices_from_nodes_rejects_invalid_connectivity(self) -> None:
        with pytest.raises(ValueError, match="triangles"):
            vertices_from_nodes(np.zeros((3, 3)), np.array([[0, 1, 3]]))


class TestLocalFrameConstruction:
    """The origin is named, never positional, so lon and lat cannot swap."""

    def test_positional_origin_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            LocalFrame(100.0, -2.0)  # type: ignore[misc]

    def test_keyword_origin(self) -> None:
        frame = LocalFrame(lon=100.0, lat=-2.0)
        assert (frame.lon, frame.lat, frame.alt) == (100.0, -2.0, 0.0)

    def test_is_exported_from_transforms_and_the_top_level(self) -> None:
        import geodef

        assert geodef.LocalFrame is LocalFrame
        assert geodef.transforms.LocalFrame is LocalFrame


class TestFlatFrame:
    """The frame flattens the Earth one way: tangent-plane East/North, height up."""

    FRAME = LocalFrame(lon=100.0, lat=40.0)

    def test_round_trip_is_exact_far_from_the_origin(self) -> None:
        east, north = np.meshgrid(np.linspace(-3e5, 3e5, 7), np.linspace(-3e5, 3e5, 7))
        geographic = self.FRAME.to_geographic(east=east, north=north, up=-12e3)
        back = self.FRAME.to_enu(
            lon=geographic[..., 0], lat=geographic[..., 1], alt=geographic[..., 2]
        )
        npt.assert_allclose(back[..., 0], east, atol=1e-6)
        npt.assert_allclose(back[..., 1], north, atol=1e-6)
        npt.assert_allclose(back[..., 2], -12e3)

    def test_depth_is_not_changed_by_distance_from_the_origin(self) -> None:
        enu = self.FRAME.to_enu(lon=[100.0, 103.0], lat=[40.0, 42.0], alt=-10e3)
        npt.assert_array_equal(enu[:, 2], [-10e3, -10e3])

    def test_surface_points_have_zero_up(self) -> None:
        enu = self.FRAME.to_enu(lon=[97.0, 103.0], lat=[38.0, 42.0])
        npt.assert_array_equal(enu[:, 2], 0.0)

    def test_north_azimuth_is_zero_at_the_origin_and_follows_convergence(self) -> None:
        azimuth = self.FRAME.north_azimuth(lon=[99.0, 100.0, 101.0], lat=40.0)
        expected = -np.array([-1.0, 0.0, 1.0]) * np.sin(np.radians(40.0))
        npt.assert_allclose(azimuth, expected, atol=1e-3)
