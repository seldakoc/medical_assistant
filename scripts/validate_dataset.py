from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_utils import index_dataset, validate_index


if __name__ == "__main__":
    index, warnings = index_dataset(Path("kaggle_3m"))
    report = validate_index(index)
    print(f"Hasta: {report['patients']}; MRI: {report['images']}")
    print(f"Eksik maske: {len(report['missing_masks'])}; okunamayan: {len(report['unreadable'])}; uyumsuz boyut: {len(report['incompatible'])}")
    print(f"Dizinleme uyarısı: {len(warnings)}")
    if report["missing_masks"] or report["unreadable"] or report["incompatible"]:
        raise SystemExit(1)
