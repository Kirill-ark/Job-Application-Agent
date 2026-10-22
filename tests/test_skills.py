from pathlib import Path

import pytest

from job_agent.skills import Skill, Taxonomy, TaxonomyError, load_taxonomy


def make_taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            Skill(name="scikit-learn", category="ml", aliases=["sklearn", "scikit learn"]),
            Skill(name="Python", category="programming"),
        ]
    )


def test_normalize_maps_alias_to_canonical_name() -> None:
    taxonomy = make_taxonomy()
    assert taxonomy.normalize("sklearn") == "scikit-learn"
    assert taxonomy.normalize("Scikit Learn") == "scikit-learn"
    assert taxonomy.normalize("  python ") == "Python"


def test_normalize_keeps_unknown_skill_as_is() -> None:
    taxonomy = make_taxonomy()
    assert taxonomy.normalize(" Rust ") == "Rust"
    assert taxonomy.category("Rust") == "other"


def test_duplicate_alias_is_an_error() -> None:
    with pytest.raises(TaxonomyError, match="used by both scikit-learn and Other"):
        Taxonomy(
            [
                Skill(name="scikit-learn", category="ml", aliases=["sklearn"]),
                Skill(name="Other", category="ml", aliases=["SKLEARN"]),
            ]
        )


def test_invalid_category_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "skills.yaml"
    path.write_text("- name: Python\n  category: snakes\n")
    with pytest.raises(TaxonomyError):
        load_taxonomy(path)


def test_project_vocabulary_loads() -> None:
    taxonomy = load_taxonomy()
    assert taxonomy.normalize("torch") == "PyTorch"
    assert taxonomy.category("PyTorch") == "ml"
