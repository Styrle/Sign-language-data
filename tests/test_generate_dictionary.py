#!/usr/bin/env python3
"""
Tests for BSL Dictionary Generator.

Tests entry generation, dictionary creation, export functions,
import/merge operations, and report generation.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_dictionary import (
    # Entry generation
    generate_sign_entry,
    load_processing_result,
    # Dictionary generation
    generate_dictionary,
    # Export functions
    dictionary_to_export_format,
    export_json,
    export_minified,
    export_typescript_types,
    # Import/merge
    import_dictionary,
    merge_dictionaries,
    diff_dictionaries,
    # Reports
    generate_coverage_report,
    generate_quality_report,
    generate_summary,
    # Validation
    validate_complete_dictionary,
)
from src.types import (
    SignDefinition,
    SignDictionaryEntry,
    SignDictionary,
    SignPose,
    ProcessingResult,
    Point3D,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_sign_definition():
    """Create a sample sign definition."""
    return SignDefinition(
        id="test_hello",
        name="Hello",
        category="greetings",
        description="A greeting sign",
        two_handed=False,
        difficulty=1,
    )


@pytest.fixture
def sample_landmarks():
    """Create sample 21-point landmarks."""
    return [Point3D(x=i * 0.05, y=i * 0.04, z=i * 0.01) for i in range(21)]


@pytest.fixture
def sample_pose(sample_landmarks):
    """Create a sample sign pose with right hand data."""
    return SignPose(
        left_hand_landmarks=None,
        right_hand_landmarks=sample_landmarks,
        tolerances={"position": 0.05, "angle": 5.0},
    )


@pytest.fixture
def sample_empty_pose():
    """Create an empty pose without landmark data."""
    return SignPose(
        left_hand_landmarks=None,
        right_hand_landmarks=None,
        tolerances={},
    )


@pytest.fixture
def sample_entry(sample_sign_definition, sample_pose):
    """Create a complete sign dictionary entry."""
    return SignDictionaryEntry(
        definition=sample_sign_definition,
        poses=[sample_pose],
        sample_count=10,
        quality_score=0.85,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        source="recorded",
    )


@pytest.fixture
def sample_dictionary(sample_entry):
    """Create a sample dictionary with one entry."""
    return SignDictionary(
        version="1.0.0",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        entries={sample_entry.definition.id: sample_entry},
        metadata={"language": "BSL"},
    )


@pytest.fixture
def multi_entry_dictionary():
    """Create a dictionary with multiple entries of varying quality."""
    entries = {}

    for i, (sign_id, cat, quality, has_data) in enumerate([
        ("bsl_alphabet_a", "alphabet", 0.9, True),
        ("bsl_alphabet_b", "alphabet", 0.7, True),
        ("bsl_colors_red", "colors", 0.5, True),
        ("bsl_colors_blue", "colors", 0.0, False),
        ("bsl_greetings_hello", "greetings", 0.85, True),
    ]):
        sign_def = SignDefinition(
            id=sign_id,
            name=sign_id.split("_")[-1].title(),
            category=cat,
            description=f"Test sign {i}",
            two_handed=False,
            difficulty=min(5, i + 1),
        )

        if has_data:
            landmarks = [Point3D(x=j * 0.05, y=j * 0.04, z=j * 0.01) for j in range(21)]
            pose = SignPose(
                right_hand_landmarks=landmarks,
                tolerances={"position": 0.05},
            )
        else:
            pose = SignPose(tolerances={})

        entry = SignDictionaryEntry(
            definition=sign_def,
            poses=[pose],
            sample_count=max(1, i * 5),
            quality_score=quality,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="recorded",
        )
        entries[sign_id] = entry

    return SignDictionary(
        version="1.0.0",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        entries=entries,
        metadata={"language": "BSL", "categories": []},
    )


# =============================================================================
# Entry Generation Tests
# =============================================================================


class TestGenerateSignEntry:
    """Tests for generate_sign_entry function."""

    def test_basic_entry_generation(self, sample_sign_definition, sample_pose):
        """Test creating a basic sign entry."""
        entry = generate_sign_entry(
            sign_def=sample_sign_definition,
            pose=sample_pose,
            quality_score=0.85,
            sample_count=10,
            source="recorded",
        )

        assert entry.definition.id == "test_hello"
        assert entry.quality_score == 0.85
        assert entry.sample_count == 10
        assert entry.source == "recorded"
        assert len(entry.poses) == 1
        assert entry.poses[0] == sample_pose

    def test_invalid_source_fallback(self, sample_sign_definition, sample_pose):
        """Test that invalid source falls back to 'merged'."""
        entry = generate_sign_entry(
            sign_def=sample_sign_definition,
            pose=sample_pose,
            source="invalid_source",
        )

        assert entry.source == "merged"

    def test_valid_sources(self, sample_sign_definition, sample_pose):
        """Test all valid source values."""
        for source in ["kaggle", "recorded", "merged"]:
            entry = generate_sign_entry(
                sign_def=sample_sign_definition,
                pose=sample_pose,
                source=source,
            )
            assert entry.source == source

    def test_timestamps_set(self, sample_sign_definition, sample_pose):
        """Test that timestamps are set to current time."""
        before = datetime.now()
        entry = generate_sign_entry(
            sign_def=sample_sign_definition,
            pose=sample_pose,
        )
        after = datetime.now()

        assert before <= entry.created_at <= after
        assert before <= entry.updated_at <= after

    def test_default_values(self, sample_sign_definition, sample_pose):
        """Test default parameter values."""
        entry = generate_sign_entry(
            sign_def=sample_sign_definition,
            pose=sample_pose,
        )

        assert entry.quality_score == 0.8
        assert entry.sample_count == 1


class TestLoadProcessingResult:
    """Tests for load_processing_result function."""

    def test_load_valid_result(self, sample_landmarks):
        """Test loading a valid processing result file."""
        result_data = {
            "canonical_pose": {
                "right_hand_landmarks": [
                    {"x": p.x, "y": p.y, "z": p.z} for p in sample_landmarks
                ],
                "tolerances": {"position": 0.05},
            },
            "quality_score": 0.9,
            "sample_count": 15,
            "warnings": ["test warning"],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result_data, f)
            filepath = Path(f.name)

        try:
            result = load_processing_result(filepath)

            assert result is not None
            assert result.quality_score == 0.9
            assert result.sample_count == 15
            assert len(result.warnings) == 1
            assert result.canonical_pose.right_hand_landmarks is not None
            assert len(result.canonical_pose.right_hand_landmarks) == 21
        finally:
            filepath.unlink()

    def test_load_with_left_hand(self, sample_landmarks):
        """Test loading result with left hand data."""
        result_data = {
            "canonical_pose": {
                "left_hand_landmarks": [
                    {"x": p.x, "y": p.y, "z": p.z} for p in sample_landmarks
                ],
                "tolerances": {},
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result_data, f)
            filepath = Path(f.name)

        try:
            result = load_processing_result(filepath)

            assert result is not None
            assert result.canonical_pose.left_hand_landmarks is not None
            assert result.canonical_pose.right_hand_landmarks is None
        finally:
            filepath.unlink()

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file returns None."""
        result = load_processing_result(Path("/nonexistent/path.json"))
        assert result is None

    def test_load_invalid_json(self):
        """Test loading invalid JSON returns None."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            filepath = Path(f.name)

        try:
            result = load_processing_result(filepath)
            assert result is None
        finally:
            filepath.unlink()

    def test_load_with_defaults(self):
        """Test loading with missing fields uses defaults."""
        result_data = {
            "canonical_pose": {"tolerances": {}},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result_data, f)
            filepath = Path(f.name)

        try:
            result = load_processing_result(filepath)

            assert result is not None
            assert result.quality_score == 0.5  # default
            assert result.sample_count == 1  # default
            assert result.warnings == []  # default
        finally:
            filepath.unlink()


# =============================================================================
# Dictionary Generation Tests
# =============================================================================


class TestGenerateDictionary:
    """Tests for generate_dictionary function."""

    def test_generate_empty_directory(self):
        """Test generating from empty processed directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dictionary = generate_dictionary(
                processed_dir=Path(tmpdir),
                version="1.0.0",
                include_missing=True,
            )

            assert dictionary.version == "1.0.0"
            # Should have entries for all SIGN_DEFINITIONS
            assert len(dictionary.entries) > 0

    def test_generate_with_processed_files(self, sample_landmarks):
        """Test generating with actual processed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a processed file
            result_data = {
                "canonical_pose": {
                    "right_hand_landmarks": [
                        {"x": p.x, "y": p.y, "z": p.z} for p in sample_landmarks
                    ],
                    "tolerances": {},
                },
                "quality_score": 0.95,
                "sample_count": 20,
            }

            # Use an existing sign ID from SIGN_DEFINITIONS
            with open(tmppath / "bsl_alphabet_a.json", "w") as f:
                json.dump(result_data, f)

            dictionary = generate_dictionary(
                processed_dir=tmppath,
                version="2.0.0",
            )

            # Check the processed entry was loaded
            if "bsl_alphabet_a" in dictionary.entries:
                entry = dictionary.entries["bsl_alphabet_a"]
                assert entry.quality_score == 0.95
                assert entry.sample_count == 20

    def test_generate_excludes_missing(self):
        """Test generate with include_missing=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dictionary = generate_dictionary(
                processed_dir=Path(tmpdir),
                version="1.0.0",
                include_missing=False,
            )

            # Should have no entries since no processed files
            assert len(dictionary.entries) == 0

    def test_generate_version_in_metadata(self):
        """Test that version is properly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dictionary = generate_dictionary(
                processed_dir=Path(tmpdir),
                version="3.2.1",
            )

            assert dictionary.version == "3.2.1"

    def test_generate_skips_processing_report(self, sample_landmarks):
        """Test that processing_report.json is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create processing_report.json (should be skipped)
            with open(tmppath / "processing_report.json", "w") as f:
                json.dump({"summary": "test"}, f)

            dictionary = generate_dictionary(
                processed_dir=tmppath,
                include_missing=False,
            )

            # Should have no entries
            assert len(dictionary.entries) == 0


# =============================================================================
# Export Function Tests
# =============================================================================


class TestDictionaryToExportFormat:
    """Tests for dictionary_to_export_format function."""

    def test_basic_export_format(self, sample_dictionary):
        """Test basic export format structure."""
        export = dictionary_to_export_format(sample_dictionary)

        assert "version" in export
        assert "created_at" in export
        assert "updated_at" in export
        assert "language" in export
        assert "sign_count" in export
        assert "signs" in export

    def test_sign_entry_flattening(self, sample_dictionary):
        """Test that sign entries are properly flattened."""
        export = dictionary_to_export_format(sample_dictionary)

        sign = export["signs"]["test_hello"]

        assert sign["id"] == "test_hello"
        assert sign["name"] == "Hello"
        assert sign["category"] == "greetings"
        assert sign["two_handed"] == False
        assert sign["difficulty"] == 1
        assert "metadata" in sign
        assert "poses" in sign

    def test_pose_export(self, sample_dictionary):
        """Test pose data in export."""
        export = dictionary_to_export_format(sample_dictionary)
        sign = export["signs"]["test_hello"]

        assert len(sign["poses"]) == 1
        pose = sign["poses"][0]

        assert "right_hand" in pose
        assert len(pose["right_hand"]) == 21
        assert "tolerances" in pose

    def test_metadata_in_export(self, sample_dictionary):
        """Test metadata fields in export."""
        export = dictionary_to_export_format(sample_dictionary)
        sign = export["signs"]["test_hello"]
        meta = sign["metadata"]

        assert meta["quality_score"] == 0.85
        assert meta["sample_count"] == 10
        assert meta["source"] == "recorded"
        assert "created_at" in meta
        assert "updated_at" in meta


class TestExportJson:
    """Tests for export_json function."""

    def test_export_pretty_json(self, sample_dictionary):
        """Test exporting formatted JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            export_json(sample_dictionary, filepath, pretty=True)

            assert filepath.exists()

            with open(filepath) as f:
                content = f.read()
                # Pretty format should have newlines
                assert "\n" in content

            data = json.loads(content)
            assert data["version"] == "1.0.0"

    def test_export_minified_json(self, sample_dictionary):
        """Test exporting minified JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.min.json"

            export_json(sample_dictionary, filepath, pretty=False)

            with open(filepath) as f:
                content = f.read()
                # Minified should have no extra whitespace
                assert "  " not in content

    def test_creates_parent_directories(self, sample_dictionary):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dirs" / "test.json"

            export_json(sample_dictionary, filepath)

            assert filepath.exists()


class TestExportTypescriptTypes:
    """Tests for export_typescript_types function."""

    def test_generates_type_file(self, sample_dictionary):
        """Test that TypeScript types are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "types.d.ts"

            export_typescript_types(sample_dictionary, filepath)

            assert filepath.exists()

    def test_sign_id_type(self, sample_dictionary):
        """Test SignId literal type generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "types.d.ts"

            export_typescript_types(sample_dictionary, filepath)

            content = filepath.read_text()

            assert "export type SignId =" in content
            assert '"test_hello"' in content

    def test_category_type(self, multi_entry_dictionary):
        """Test SignCategory type includes all categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "types.d.ts"

            export_typescript_types(multi_entry_dictionary, filepath)

            content = filepath.read_text()

            assert "export type SignCategory =" in content
            assert '"alphabet"' in content
            assert '"colors"' in content
            assert '"greetings"' in content

    def test_interface_definitions(self, sample_dictionary):
        """Test that all interfaces are defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "types.d.ts"

            export_typescript_types(sample_dictionary, filepath)

            content = filepath.read_text()

            assert "export interface Point3D" in content
            assert "export interface HandPose" in content
            assert "export interface SignPose" in content
            assert "export interface SignEntry" in content
            assert "export interface BSLDictionary" in content

    def test_sign_count_constant(self, multi_entry_dictionary):
        """Test SIGN_COUNT constant is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "types.d.ts"

            export_typescript_types(multi_entry_dictionary, filepath)

            content = filepath.read_text()

            # Should have 5 signs
            assert "export const SIGN_COUNT: 5" in content


# =============================================================================
# Import/Merge Tests
# =============================================================================


class TestImportDictionary:
    """Tests for import_dictionary function."""

    def test_import_exported_dictionary(self, sample_dictionary):
        """Test roundtrip export/import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            export_json(sample_dictionary, filepath)
            imported = import_dictionary(filepath)

            assert imported is not None
            assert imported.version == "1.0.0"
            assert "test_hello" in imported.entries

    def test_import_nonexistent_file(self):
        """Test importing non-existent file returns None."""
        result = import_dictionary(Path("/nonexistent/dict.json"))
        assert result is None

    def test_import_preserves_data(self, multi_entry_dictionary):
        """Test that import preserves all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            export_json(multi_entry_dictionary, filepath)
            imported = import_dictionary(filepath)

            assert len(imported.entries) == 5

            # Check a specific entry
            entry = imported.entries["bsl_alphabet_a"]
            assert entry.definition.category == "alphabet"
            assert entry.quality_score == 0.9


class TestMergeDictionaries:
    """Tests for merge_dictionaries function."""

    def test_merge_adds_new_entries(self):
        """Test merging adds entries from additions."""
        base_def = SignDefinition(
            id="sign_a", name="A", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )
        add_def = SignDefinition(
            id="sign_b", name="B", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )

        base_entry = SignDictionaryEntry(
            definition=base_def,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )
        add_entry = SignDictionaryEntry(
            definition=add_def,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )

        base = SignDictionary(
            version="1.0.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_a": base_entry},
            metadata={},
        )
        additions = SignDictionary(
            version="1.1.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_b": add_entry},
            metadata={},
        )

        merged = merge_dictionaries(base, additions)

        assert len(merged.entries) == 2
        assert "sign_a" in merged.entries
        assert "sign_b" in merged.entries

    def test_merge_prefers_higher_quality(self):
        """Test that merge keeps higher quality entry by default."""
        sign_def = SignDefinition(
            id="sign_a", name="A", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )

        low_entry = SignDictionaryEntry(
            definition=sign_def,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.5,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )
        high_entry = SignDictionaryEntry(
            definition=sign_def,
            poses=[SignPose(tolerances={})],
            sample_count=10, quality_score=0.9,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )

        base = SignDictionary(
            version="1.0.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_a": low_entry},
            metadata={},
        )
        additions = SignDictionary(
            version="1.1.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_a": high_entry},
            metadata={},
        )

        merged = merge_dictionaries(base, additions, prefer_higher_quality=True)

        assert merged.entries["sign_a"].quality_score == 0.9

    def test_merge_version_increment(self):
        """Test that merged version is incremented."""
        base = SignDictionary(
            version="1.5.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={},
            metadata={},
        )
        additions = SignDictionary(
            version="1.6.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={},
            metadata={},
        )

        merged = merge_dictionaries(base, additions)

        assert merged.version == "1.6.0"


class TestDiffDictionaries:
    """Tests for diff_dictionaries function."""

    def test_diff_detects_added(self):
        """Test diff detects added entries."""
        sign_def = SignDefinition(
            id="new_sign", name="New", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )
        new_entry = SignDictionaryEntry(
            definition=sign_def,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )

        old = SignDictionary(
            version="1.0.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={},
            metadata={},
        )
        new = SignDictionary(
            version="1.1.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"new_sign": new_entry},
            metadata={},
        )

        diff = diff_dictionaries(old, new)

        assert "new_sign" in diff["added"]
        assert diff["summary"]["total_added"] == 1

    def test_diff_detects_removed(self):
        """Test diff detects removed entries."""
        sign_def = SignDefinition(
            id="old_sign", name="Old", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )
        old_entry = SignDictionaryEntry(
            definition=sign_def,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )

        old = SignDictionary(
            version="1.0.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"old_sign": old_entry},
            metadata={},
        )
        new = SignDictionary(
            version="1.1.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={},
            metadata={},
        )

        diff = diff_dictionaries(old, new)

        assert "old_sign" in diff["removed"]
        assert diff["summary"]["total_removed"] == 1

    def test_diff_detects_quality_improvement(self):
        """Test diff detects quality improvements."""
        sign_def = SignDefinition(
            id="sign_a", name="A", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )

        old_entry = SignDictionaryEntry(
            definition=sign_def,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.5,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )
        new_entry = SignDictionaryEntry(
            definition=sign_def,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.9,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )

        old = SignDictionary(
            version="1.0.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_a": old_entry},
            metadata={},
        )
        new = SignDictionary(
            version="1.1.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_a": new_entry},
            metadata={},
        )

        diff = diff_dictionaries(old, new)

        assert "sign_a" in diff["quality_improved"]
        assert diff["summary"]["quality_improvements"] == 1


# =============================================================================
# Report Tests
# =============================================================================


class TestGenerateCoverageReport:
    """Tests for generate_coverage_report function."""

    def test_report_contains_header(self, multi_entry_dictionary):
        """Test report has proper header."""
        report = generate_coverage_report(multi_entry_dictionary)

        assert "BSL Dictionary Coverage Report" in report
        assert "Version: 1.0.0" in report

    def test_report_shows_overall_coverage(self, multi_entry_dictionary):
        """Test overall coverage is calculated."""
        report = generate_coverage_report(multi_entry_dictionary)

        # 4 out of 5 have data = 80%
        assert "Overall Coverage:" in report
        assert "80.0%" in report

    def test_report_shows_category_breakdown(self, multi_entry_dictionary):
        """Test category breakdown is shown."""
        report = generate_coverage_report(multi_entry_dictionary)

        assert "Coverage by Category:" in report
        assert "alphabet" in report
        assert "colors" in report
        assert "greetings" in report

    def test_report_lists_missing_signs(self, multi_entry_dictionary):
        """Test missing signs are listed."""
        report = generate_coverage_report(multi_entry_dictionary)

        assert "Missing Data" in report
        assert "Blue" in report  # bsl_colors_blue has no data


class TestGenerateQualityReport:
    """Tests for generate_quality_report function."""

    def test_report_contains_header(self, multi_entry_dictionary):
        """Test quality report has header."""
        report = generate_quality_report(multi_entry_dictionary)

        assert "BSL Dictionary Quality Report" in report

    def test_report_shows_average_quality(self, multi_entry_dictionary):
        """Test average quality is shown."""
        report = generate_quality_report(multi_entry_dictionary)

        assert "Average Quality Score:" in report

    def test_report_shows_quality_distribution(self, multi_entry_dictionary):
        """Test quality distribution breakdown."""
        report = generate_quality_report(multi_entry_dictionary)

        assert "Quality Distribution:" in report
        assert "High" in report
        assert "Medium" in report
        assert "Low" in report

    def test_report_shows_sample_stats(self, multi_entry_dictionary):
        """Test sample statistics are shown."""
        report = generate_quality_report(multi_entry_dictionary)

        assert "Sample Statistics:" in report
        assert "Total Samples:" in report


class TestGenerateSummary:
    """Tests for generate_summary function."""

    def test_summary_structure(self, multi_entry_dictionary):
        """Test summary has all required fields."""
        summary = generate_summary(multi_entry_dictionary)

        assert "version" in summary
        assert "total_signs" in summary
        assert "signs_with_data" in summary
        assert "coverage_percent" in summary
        assert "quality" in summary
        assert "samples" in summary
        assert "by_category" in summary
        assert "by_source" in summary

    def test_summary_counts(self, multi_entry_dictionary):
        """Test summary counts are correct."""
        summary = generate_summary(multi_entry_dictionary)

        assert summary["total_signs"] == 5
        assert summary["signs_with_data"] == 4
        assert summary["coverage_percent"] == 80.0

    def test_summary_by_category(self, multi_entry_dictionary):
        """Test category breakdown in summary."""
        summary = generate_summary(multi_entry_dictionary)

        by_cat = summary["by_category"]
        assert by_cat["alphabet"] == 2
        assert by_cat["colors"] == 2
        assert by_cat["greetings"] == 1

    def test_summary_quality_stats(self, multi_entry_dictionary):
        """Test quality statistics in summary."""
        summary = generate_summary(multi_entry_dictionary)

        quality = summary["quality"]
        assert quality["max"] == 0.9
        assert quality["min"] == 0.0
        assert quality["high_quality_count"] == 2  # 0.9 and 0.85


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateCompleteDictionary:
    """Tests for validate_complete_dictionary function."""

    def test_validates_all_entries(self, multi_entry_dictionary):
        """Test all entries are validated."""
        result = validate_complete_dictionary(multi_entry_dictionary)

        assert "entry_results" in result
        assert len(result["entry_results"]) == 5

    def test_returns_error_count(self, multi_entry_dictionary):
        """Test error count is returned."""
        result = validate_complete_dictionary(multi_entry_dictionary)

        assert "error_count" in result
        assert "warning_count" in result
        assert "is_valid" in result

    def test_empty_dictionary_valid(self):
        """Test empty dictionary is valid."""
        empty = SignDictionary(
            version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            entries={},
            metadata={},
        )

        result = validate_complete_dictionary(empty)

        assert result["is_valid"] == True
        assert result["error_count"] == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullWorkflow:
    """End-to-end workflow tests."""

    def test_generate_export_import_roundtrip(self, sample_landmarks):
        """Test complete generate -> export -> import roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            output_dir = Path(tmpdir) / "output"
            processed_dir.mkdir()
            output_dir.mkdir()

            # Create a processed file
            result_data = {
                "canonical_pose": {
                    "right_hand_landmarks": [
                        {"x": p.x, "y": p.y, "z": p.z} for p in sample_landmarks
                    ],
                    "tolerances": {"position": 0.05},
                },
                "quality_score": 0.88,
                "sample_count": 12,
            }

            with open(processed_dir / "bsl_alphabet_a.json", "w") as f:
                json.dump(result_data, f)

            # Generate dictionary
            dictionary = generate_dictionary(
                processed_dir=processed_dir,
                version="1.0.0",
            )

            # Export
            json_path = output_dir / "dict.json"
            export_json(dictionary, json_path)

            # Import
            imported = import_dictionary(json_path)

            assert imported is not None
            assert imported.version == "1.0.0"

            # Check entry was properly round-tripped
            if "bsl_alphabet_a" in imported.entries:
                entry = imported.entries["bsl_alphabet_a"]
                assert abs(entry.quality_score - 0.88) < 0.01

    def test_merge_and_diff_workflow(self):
        """Test merging dictionaries and computing diff."""
        sign_def_a = SignDefinition(
            id="sign_a", name="A", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )
        sign_def_b = SignDefinition(
            id="sign_b", name="B", category="alphabet",
            description="", two_handed=False, difficulty=1,
        )

        entry_a = SignDictionaryEntry(
            definition=sign_def_a,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.7,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )
        entry_b = SignDictionaryEntry(
            definition=sign_def_b,
            poses=[SignPose(tolerances={})],
            sample_count=5, quality_score=0.8,
            created_at=datetime.now(), updated_at=datetime.now(),
            source="recorded",
        )

        base = SignDictionary(
            version="1.0.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_a": entry_a},
            metadata={},
        )
        additions = SignDictionary(
            version="1.1.0",
            created_at=datetime.now(), updated_at=datetime.now(),
            entries={"sign_b": entry_b},
            metadata={},
        )

        # Merge
        merged = merge_dictionaries(base, additions)

        # Compute diff from base to merged
        diff = diff_dictionaries(base, merged)

        assert diff["summary"]["total_added"] == 1
        assert "sign_b" in diff["added"]
