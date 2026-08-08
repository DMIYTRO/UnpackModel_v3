import re
from pathlib import Path


# Сначала проверяются эти дешёвые словари для трёх основных языков.
SIDE_MARKERS = {
    "face": {
        "ru": (
            "лицо", "лицевая", "лиц", "перед", "передняя", "титул",
            "титульная", "фас", "морда", "обложка", "лицевая сторона",
            "1 сторона", "первая сторона", "главная", "фронт",
        ),
        "uk": (
            "лице", "лицьова", "лиц", "перед", "передня", "титул",
            "титульна", "обкладинка", "лицьова сторона", "1 сторона",
            "перша сторона", "головна", "фронт",
        ),
        "en": (
            "front", "face", "cover", "obverse", "recto", "main",
            "primary", "side1", "side_1", "page1", "page_1", "top",
            "outer", "outside",
        ),
    },
    "back": {
        "ru": (
            "оборот", "оборотная", "обор", "зад", "задняя", "тыл",
            "тыльная", "изнанка", "оборотная сторона", "2 сторона",
            "вторая сторона", "обратная",
        ),
        "uk": (
            "зворот", "зворотна", "звор", "зад", "задня", "тил",
            "тильна", "виворіт", "зворотна сторона", "2 сторона",
            "друга сторона", "обернена",
        ),
        "en": (
            "back", "reverse", "verso", "rear", "flip", "side2",
            "side_2", "page2", "page_2", "bottom", "secondary", "inner",
            "inside",
        ),
    },
}

# Сокращения проверяются только как самостоятельные токены. Поиск одной буквы
# как подстроки сделал бы почти любое имя ложным совпадением.
SIDE_ABBREVIATIONS = {
    "face": ("f", "фас", "л", "прд", "cvr", "frt", "1f", "f1"),
    "back": ("b", "звр", "об", "зд", "rvs", "bck", "2b", "b2"),
}

# Маркеры окружены границами слов, чтобы, например, ``лице`` не совпадало
# внутри слова ``таблице``. Список покрывает распространённые русские,
# украинские и английские названия сторон.
FACE_WORD_PATTERNS = (
    r"(?<!\w)лиц(?:о|е|евой|евая|евое|евые|евой|евої|ева|евий|еві)(?!\w)",
    r"(?<!\w)лицьов(?:а|ий|е|і|ої|ому)?(?!\w)",
    r"(?<!\w)(?:перед|передня|передняя|передній|передний|фасад|титул)(?!\w)",
    r"(?<!\w)(?:face|front|recto|cover)(?!\w)",
    r"(?<!\w)(?:сторона|side)[\s._-]*(?:1|a|а)(?!\w)",
    r"(?<!\w)(?:1|a|а)[\s._-]*(?:сторона|side)(?!\w)",
)
FACE_PATTERNS = (
    r"(?:^|[\s._-])(?:1|f|a|л)$",
    r"(?:\(1\)|#1)(?!\d)",
)
BACK_WORD_PATTERNS = (
    r"(?<!\w)оборот(?:ная|ний|на|ное|ний|ка)?(?!\w)",
    r"(?<!\w)обрат(?:ная|ный|на|ний|ное|ка)(?!\w)",
    r"(?<!\w)зворот(?:на|ний|ній|не|ній|ка)?(?!\w)",
    r"(?<!\w)(?:тыл|тильна|тыльная)(?!\w)",
    r"(?<!\w)(?:back|rear|verso|reverse)(?!\w)",
    r"(?<!\w)(?:сторона|side)[\s._-]*(?:2|b|б)(?!\w)",
    r"(?<!\w)(?:2|b|б)[\s._-]*(?:сторона|side)(?!\w)",
    r"back$",
)
BACK_PATTERNS = (
    r"(?:^|[\s._-])(?:2|b|z|о|з)$",
    r"(?:\(2\)|#2)(?!\d)",
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


def _side_matches(texts: list[str]) -> tuple[bool, bool]:
    raw = " ".join(texts).casefold()
    normalized = re.sub(r"[._-]+", " ", raw)

    def has_marker(side: str) -> bool:
        markers = {
            marker.casefold().replace("_", " ")
            for language_markers in SIDE_MARKERS[side].values()
            for marker in language_markers
        }
        for marker in markers:
            phrase = r"[\s._-]+".join(
                re.escape(part) for part in marker.split()
            )
            if re.search(rf"(?<!\w){phrase}(?!\w)", normalized):
                return True

            # Длинные характерные обозначения поддерживаются также внутри
            # составного имени без разделителей: ``cardfrontfinal``.
            compact_marker = re.sub(r"[\s._-]+", "", marker)
            compact_name = re.sub(r"[\s._-]+", "", raw)
            if (
                len(compact_marker) >= 5
                and not any(character.isdigit() for character in compact_marker)
                and compact_marker in compact_name
            ):
                return True
        return False

    def has_abbreviation(side: str) -> bool:
        return any(
            re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", normalized)
            for marker in SIDE_ABBREVIATIONS[side]
        )

    # Явные слова имеют приоритет над слабыми числовыми маркерами. Например,
    # ``оборот (1)`` — это оборот с номером копии, а не лицевая сторона.
    face_word = has_marker("face") or any(
        re.search(pattern, normalized) for pattern in FACE_WORD_PATTERNS
    )
    back_word = has_marker("back") or any(
        re.search(pattern, normalized) for pattern in BACK_WORD_PATTERNS
    )
    if face_word or back_word:
        return face_word, back_word

    return (
        has_abbreviation("face")
        or any(re.search(pattern, raw) for pattern in FACE_PATTERNS),
        has_abbreviation("back")
        or any(re.search(pattern, raw) for pattern in BACK_PATTERNS),
    )


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
        stem = stem_getter(item)
        is_face, is_back = _side_matches([stem])
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
    # Для пары одного явного маркера достаточно: второй файл может быть только
    # противоположной стороной. Это также предотвращает ненужный вызов LLM.
    if len(items) == 2:
        unmatched = [
            item for item in items if item not in face_candidates + back_candidates
        ]
        if len(unmatched) == 1:
            if result["face"] is not None and result["back"] is None:
                result["back"] = unmatched[0]
            elif result["back"] is not None and result["face"] is None:
                result["face"] = unmatched[0]
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
