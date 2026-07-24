import re
from pathlib import Path

# Маркеры окружены границами слов, чтобы, например, ``лице`` не совпадало
# внутри слова ``таблице``. Список покрывает распространённые русские,
# украинские и английские названия сторон.
FACE_PATTERNS = (
    r"(?<!\w)лиц(?:о|е|евой|евая|евое|евые|евой|евої|ева|евий|еві)(?!\w)",
    r"(?<!\w)лицьов(?:а|ий|е|і|ої|ому)?(?!\w)",
    r"(?<!\w)(?:перед|передня|передняя|передній|передний|фасад|титул)(?!\w)",
    r"(?<!\w)(?:face|front|recto)(?!\w)",
    r"(?<!\w)(?:сторона|side)[\s._-]*(?:1|a|а)(?!\w)",
    r"(?<!\w)(?:1|a|а)[\s._-]*(?:сторона|side)(?!\w)",
    r"^1$",
)
BACK_PATTERNS = (
    r"(?<!\w)оборот(?:ная|ний|на|ное|ний|ка)?(?!\w)",
    r"(?<!\w)обрат(?:ная|ный|на|ний|ное|ка)(?!\w)",
    r"(?<!\w)зворот(?:на|ний|ній|не|ній|ка)?(?!\w)",
    r"(?<!\w)(?:тыл|тильна|тыльная)(?!\w)",
    r"(?<!\w)(?:back|rear|verso)(?!\w)",
    r"(?<!\w)(?:сторона|side)[\s._-]*(?:2|b|б)(?!\w)",
    r"(?<!\w)(?:2|b|б)[\s._-]*(?:сторона|side)(?!\w)",
    r"^2$",
)

# Вспомогательные изображения не являются печатными сторонами. Наряду с
# правильными вариантами здесь перечислены частые перестановки и пропуски букв.
AUXILIARY_PATTERNS = (
    r"(?<!\w)(?:превью|превю|првеью|первью|прев(?:ь)?юшка)(?!\w)",
    r"(?<!\w)(?:preview|prewiew|prevew|previev|prevju|prevyu)(?!\w)",
    r"(?<!\w)(?:макет|layout)[\s._-]*(?:для[\s._-]*)?(?:согласования|approval)(?!\w)",
)


def is_auxiliary_filename(filename: str) -> bool:
    """Возвращает True для явно обозначенного превью/файла согласования."""
    stem = Path(filename).stem.casefold().replace("_", " ")
    return any(re.search(pattern, stem) for pattern in AUXILIARY_PATTERNS)


def _classify(items: list, stem_getter) -> dict:
    """Классифицирует стороны только при однозначном совпадении.

    Несколько кандидатов на одну сторону, совпадение обеих сторон в одном
    имени и отсутствие явных признаков считаются неоднозначностью. В таком
    случае незаполненные стороны остаются ``None`` и вызывающий код может
    направить заказ оператору, не угадывая порядок файлов.
    """
    face_candidates = []
    back_candidates = []
    contradictory = False

    for item in items:
        # В клиентских именах подчёркивание обычно является разделителем слов,
        # хотя регулярные выражения считают его ``\w``-символом.
        stem = stem_getter(item).casefold().replace("_", " ")
        is_face = any(re.search(pattern, stem) for pattern in FACE_PATTERNS)
        is_back = any(re.search(pattern, stem) for pattern in BACK_PATTERNS)
        if is_face and is_back:
            contradictory = True
        elif is_face:
            face_candidates.append(item)
        elif is_back:
            back_candidates.append(item)

    if contradictory or len(face_candidates) > 1 or len(back_candidates) > 1:
        return {"face": None, "back": None}

    result = {
        "face": face_candidates[0] if face_candidates else None,
        "back": back_candidates[0] if back_candidates else None,
    }
    # Единственный файл одностороннего заказа по-прежнему является лицом.
    if len(items) == 1 and not result["face"] and not result["back"]:
        result["face"] = items[0]
    if result["face"] is not None and result["face"] == result["back"]:
        return {"face": None, "back": None}
    return result


def classify_face_back(filenames: list) -> dict:
    """
    Принимает список имен файлов (например, для папки 4-4) 
    и распределяет их на 'лицо' и 'оборот' на основе словаря ключевых слов.
    """
    return _classify(filenames, lambda item: item.rsplit(".", 1)[0])


def classify_face_back_paths(files: list[Path]) -> dict:
    """Версия классификатора для файлов из подпапок.

    Сторона определяется по имени самого файла, а не по имени подпапки.
    Возвращаются объекты Path, поэтому одинаковые имена в разных подпапках
    не смешиваются.
    """
    return _classify(files, lambda item: item.stem)

if __name__ == "__main__":
    # Тест на реальной папке с 4-4
    test_dir = Path("test_archives/04_NP_Glam11_350_mel_(90x50)_4-4_T100_(17618-25516399)_offset-face")
    if test_dir.exists():
        files = [f.name for f in test_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        print(f"Файлы в папке: {files}")
        classified = classify_face_back(files)
        print(f"Результат распределения: Лицо -> {classified['face']}, Оборот -> {classified['back']}\n")
    
    # Синтетические тесты
    test_cases = [
        ['1.pdf', '2.pdf'],
        ['face_image.tif', 'back_image.tif'],
        ['непонятно_что.tif', 'оборот.tif']
    ]
    
    for case in test_cases:
        classified = classify_face_back(case)
        print(f"Тест {case}: Лицо -> {classified['face']}, Оборот -> {classified['back']}")
