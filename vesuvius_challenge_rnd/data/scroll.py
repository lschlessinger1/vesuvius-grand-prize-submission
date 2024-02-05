import json
from functools import cached_property
from pathlib import Path

import pint

from vesuvius_challenge_rnd import SCROLL_DATA_DIR, ureg
from vesuvius_challenge_rnd.data.volumetric_segment import VolumetricSegment

VOXEL_SIZE_MICRONS = 7.91
VOXEL_SIZE = VOXEL_SIZE_MICRONS * ureg.micron
MONSTER_SEGMENT_PREFIX = "Scroll1_part_1_wrap"


class ScrollSegment(VolumetricSegment):
    """A segment of a scroll."""

    surface_volume_dir_name = "layers"

    def __init__(
        self,
        scroll_id: str,
        segment_name: str,
        scroll_dir: Path = SCROLL_DATA_DIR,
    ):
        """
        Initializes a ScrollSegment instance.

        Args:
            scroll_id (str): The unique identifier for the scroll.
            segment_name (str): The name of the segment.
            scroll_dir (Path, optional): The directory where the scroll data is located.
                Defaults to SCROLL_DATA_DIR.
        """
        data_dir = scroll_dir / scroll_id
        # Infer superseded.
        self.segment_name_orig = segment_name
        seg_name_parts = self.segment_name_orig.split("_superseded")
        self.is_superseded = len(seg_name_parts) == 2
        self.is_subsegment = len(self.segment_name_orig.split("_C")) == 2
        segment_name = seg_name_parts[0]
        super().__init__(data_dir, segment_name)
        self.scroll_id = scroll_id
        self.scroll_dir = scroll_dir

    @property
    def papyrus_mask_file_name(self) -> str:
        """The file name for the papyrus mask of the segment.

        Returns:
            str: The file name for the papyrus mask.
        """
        if not self.is_subsegment:
            return f"{self.segment_name}_mask.png"
        else:
            return f"{self.segment_name_orig}_mask.png"

    @property
    def voxel_size_microns(self) -> float:
        """Get the voxel size in microns.

        Returns:
            float: The voxel size in microns.
        """
        return VOXEL_SIZE_MICRONS

    @property
    def voxel_size(self) -> pint.Quantity:
        """Get the voxel size as a pint quantity.

        Returns:
            pint.Quantity: The voxel size, using microns as the unit.
        """
        return VOXEL_SIZE

    @cached_property
    def author(self) -> str:
        """Get the annotator of the scroll segment.

        Returns:
            str: The name or identifier of the annotator.
        """
        with open(self.volume_dir_path / "author.txt") as f:
            author = f.read()
        return author

    @cached_property
    def area_cm2(self) -> float:
        """Get the area of the scroll segment in units of centimeters squared.

        Returns:
            float: The area of the scroll segment.
        """
        with open(self.volume_dir_path / "area_cm2.txt") as f:
            area_cm2 = float(f.read())
        return area_cm2

    @property
    def volume_dir_path(self) -> Path:
        """The volumetric segment data directory path.

        Returns:
            Path: Path to the volume directory.
        """
        if not self.is_superseded:
            return super().volume_dir_path
        else:
            return self.data_dir / self.segment_name_orig

    @cached_property
    def metadata(self) -> dict[str, str]:
        """Retrieve the metadata for the scroll segment.

        Returns:
            dict[str, str]: A dictionary containing the metadata for the scroll segment.
        """
        with open(self.volume_dir_path / "meta.json") as f:
            metadata = json.load(f)
        return metadata

    @property
    def ppm_path(self) -> Path:
        return self.volume_dir_path / f"{self.segment_name}.ppm"

    @property
    def mesh_path(self) -> Path:
        return self.volume_dir_path / f"{self.segment_name}.obj"

    def __repr__(self):
        return f"{self.__class__.__name__}(scroll_id={self.scroll_id}, segment_name={self.segment_name}, shape={self.shape})"


class MonsterSegment(ScrollSegment):
    """A specific type of scroll segment, known as a MonsterSegment."""

    surface_volume_dir_name = "surface_volume"

    def __init__(
        self,
        segment_name: str,
        scroll_dir: Path = SCROLL_DATA_DIR,
    ):
        """
        Initializes a MonsterSegment instance.

        Args:
            segment_name (str): The name of the segment.
            scroll_dir (Path, optional): The directory where the scroll data is located.
                Defaults to SCROLL_DATA_DIR.
        """
        super().__init__(scroll_id="1", segment_name=segment_name, scroll_dir=scroll_dir)
        self._orientation = self.segment_name.split("_")[-1]

    @property
    def orientation(self) -> str:
        """Get the orientation of the MonsterSegment.

        Returns:
            str: The orientation of the MonsterSegment.
        """
        return self._orientation

    @classmethod
    def from_orientation(cls, orientation: str, scroll_dir: Path = SCROLL_DATA_DIR):
        """Create a MonsterSegment from a specific orientation.

        Args:
            orientation (str): The orientation for the MonsterSegment.
            scroll_dir (Path, optional): The directory where the scroll data is located.
                Defaults to SCROLL_DATA_DIR.

        Returns:
            MonsterSegment: The MonsterSegment instance.
        """
        segment_name = f"{MONSTER_SEGMENT_PREFIX}_{orientation}"
        return cls(segment_name, scroll_dir=scroll_dir)

    @cached_property
    def author(self) -> str:
        """
        Get the annotator of the scroll segment.

        Returns:
            str: The name of the annotator.
        """
        return "stephen"

    @cached_property
    def area_cm2(self) -> float:
        """
        Get the area of the scroll segment in centimeters squared.

        Returns:
            float: The area of the segment, calculated using the mask and voxel size.
        """
        seg_mask = self.load_mask()
        return (seg_mask.sum() * self.voxel_size_microns**2) / 1e8

    @cached_property
    def metadata(self) -> dict[str, str]:
        """
        Get the metadata of the segment.

        Returns:
            dict[str, str]: The metadata, loaded from a JSON file.
        """
        surface_volume_dir = self.volume_dir_path / self.surface_volume_dir_name
        with open(surface_volume_dir / "meta.json") as f:
            metadata = json.load(f)
        return metadata


class MonsterSegmentVerso(MonsterSegment):
    """
    Represents the Verso orientation (reverse side) of a monster scroll segment.
    This class encapsulates specific behavior or properties associated with the reverse side of the segment.
    """

    def __init__(
        self,
        segment_name: str = f"{MONSTER_SEGMENT_PREFIX}_verso",
        scroll_dir: Path = SCROLL_DATA_DIR,
    ):
        """
        Initialize a Verso (reverse side) monster scroll segment.
        Args:
            segment_name (str, optional): The name of the segment. Defaults to f"{MONSTER_SEGMENT_PREFIX}_verso".
            scroll_dir (Path, optional): The directory path containing the scroll data. Defaults to SCROLL_DATA_DIR.
        """
        super().__init__(segment_name, scroll_dir=scroll_dir)


class MonsterSegmentRecto(MonsterSegment):
    """
    Represents the Recto orientation (front side) of a monster scroll segment.
    This class encapsulates specific behavior or properties associated with the front side of the segment.
    """

    def __init__(
        self,
        segment_name: str = f"{MONSTER_SEGMENT_PREFIX}_recto",
        scroll_dir: Path = SCROLL_DATA_DIR,
    ):
        """
        Initialize a Recto (front side) monster scroll segment.
        Args:
            segment_name (str, optional): The name of the segment. Defaults to f"{MONSTER_SEGMENT_PREFIX}_recto".
            scroll_dir (Path, optional): The directory path containing the scroll data. Defaults to SCROLL_DATA_DIR.
        """
        super().__init__(segment_name, scroll_dir=scroll_dir)


class Scroll:
    """A collection of scroll segments."""

    def __init__(self, scroll_id: str, scroll_dir: Path = SCROLL_DATA_DIR):
        """
        Initializes a Scroll instance representing a collection of scroll segments.

        Args:
            scroll_id (str): The unique identifier for the scroll.
            scroll_dir (Path, optional): The directory where the scroll data is located.
                Defaults to SCROLL_DATA_DIR.
        """
        self.scroll_id = scroll_id
        self.scroll_dir = scroll_dir
        self.data_dir = self.scroll_dir / str(self.scroll_id)

        self.segments: list[ScrollSegment] = []
        self.missing_segment_names: list[str] = []
        for segment_path in self.data_dir.glob("*"):
            segment_name = segment_path.name
            try:
                segment = create_scroll_segment(self.scroll_id, segment_name, self.scroll_dir)
                self.segments.append(segment)
            except ValueError:
                self.missing_segment_names.append(segment_name)

    @property
    def segment_names(self) -> list[str]:
        """Get the segment names of the scroll.

        Returns:
            list[str]: A list of segment names.
        """
        return [segment.segment_name for segment in self.segments]

    @property
    def n_missing_segments(self) -> int:
        """Get the number of scroll segments missing data.

        Returns:
            int: The number of missing segments.
        """
        return len(self.missing_segment_names)

    @property
    def voxel_size_microns(self) -> float:
        """Get the voxel size in microns.

        Returns:
            float: The voxel size in microns.
        """
        return VOXEL_SIZE_MICRONS

    @property
    def voxel_size(self) -> pint.Quantity:
        """Get the voxel size as a pint quantity.

        Returns:
            pint.Quantity: The voxel size, using microns as the unit.
        """
        return VOXEL_SIZE

    def __getitem__(self, index: int) -> ScrollSegment:
        """Get a scroll segment by its index within the collection.

        Args:
            index (int): The index of the segment.

        Returns:
            ScrollSegment: The scroll segment at the given index.
        """
        return self.segments[index]

    def __len__(self) -> int:
        """Get the number of segments in the scroll.

        Returns:
            int: The number of segments.
        """
        return len(self.segments)

    def __repr__(self):
        return f"{self.__class__.__name__}(scroll_id={self.scroll_id}, num segments={len(self)})"


def create_scroll_segment(
    scroll_id: str, segment_name: str, scroll_dir: Path = SCROLL_DATA_DIR
) -> ScrollSegment:
    """
    Creates a scroll segment object based on the given parameters.

    Args:
        scroll_id (str): The unique identifier for the scroll.
        segment_name (str): The name of the segment to be created.
        scroll_dir (Path, optional): The directory where scroll data is stored. Defaults to SCROLL_DATA_DIR.

    Returns:
        ScrollSegment: The created scroll segment object.

    Raises:
        ValueError: If no surface volumes are found for the given segment.
    """
    segment_path = build_scroll_segment_path(scroll_dir, scroll_id, segment_name)
    is_not_monster_segment = not segment_name.startswith(MONSTER_SEGMENT_PREFIX)
    shared_segment_kwargs = {"segment_name": segment_name, "scroll_dir": scroll_dir}
    if is_not_monster_segment:
        segment_type = ScrollSegment
        segment_kwargs = {"scroll_id": scroll_id} | shared_segment_kwargs
    else:
        segment_type = MonsterSegment
        segment_kwargs = shared_segment_kwargs

    surface_vol_dir_path = segment_path / segment_type.surface_volume_dir_name
    if surface_vol_dir_path.is_dir():
        segment = segment_type(**segment_kwargs)
    else:
        raise ValueError(
            f"Could not create scroll segment {segment_name}. No surface volumes found."
        )
    return segment


def build_scroll_segment_path(scroll_dir: Path, scroll_id: str, segment_name: str) -> Path:
    """
    Builds the file path for a scroll segment.

    Args:
        scroll_dir (Path): The directory where scroll data is stored.
        scroll_id (str): The unique identifier for the scroll.
        segment_name (str): The name of the segment.

    Returns:
        Path: The constructed file path for the segment.
    """
    return scroll_dir / scroll_id / segment_name
