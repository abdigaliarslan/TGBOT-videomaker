"""Поведение режиссёра вокруг LLM: приём, один ретрай, фолбэк (§3).

Ключевое требование: что бы модель ни вернула, наружу выходит валидный EDL.
"""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.edl import Word, validate_edl
from src.llm.client import LLMClient
from src.pipeline.director import clean_transcript, direct
from tests.conftest import FakeProvider, result_of


def cfg() -> Settings:
    return Settings(
        _env_file=None,
        sub_max_words=4, sub_max_chars=40, sub_gap_split=0.6, cut_max_gap=0.8,
        llm_provider="deepseek", deepseek_api_key="test",
    )


def words(n: int = 9) -> list[Word]:
    return [Word(idx=i, text=f"сл{i}", start=i * 0.4, end=i * 0.4 + 0.3) for i in range(n)]


def good_plan() -> dict:
    return {
        "hook_end_word": 2,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1, 2], "register": "poster",
             "accent_word": 1, "accent": "accent", "position": "auto", "follow_face": False},
            {"words": [3, 4, 5], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [6, 7, 8], "register": "poster",
             "accent_word": 8, "accent": "shout", "position": "auto", "follow_face": False},
        ],
    }


def broken_plan() -> dict:
    """Слова 4 и 5 потеряны — валидатор обязан это поймать."""
    return {
        "hook_end_word": None,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1, 2], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [3, 6, 7, 8], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
        ],
    }


def client_with(*payloads) -> LLMClient:
    provider = FakeProvider([result_of(p) for p in payloads])
    c = LLMClient(cfg=cfg(), provider=provider)
    c.fake = provider  # type: ignore[attr-defined]
    return c


async def test_good_plan_is_accepted():
    ws = words()
    llm = client_with(good_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == list(range(9))
    assert any(cw.emphasis.value == "shout" for b in edl.captions for cw in b.words)
    assert len(llm.fake.calls) == 1


async def test_broken_plan_triggers_exactly_one_retry():
    ws = words()
    llm = client_with(broken_plan(), good_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 2, "должен быть ровно один ретрай"


async def test_retry_carries_the_validation_error():
    ws = words()
    llm = client_with(broken_plan(), good_plan())
    await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    second = llm.fake.calls[1]["user"]
    assert "Предыдущий ответ не подошёл" in second
    assert "не попали ни в один блок" in second


async def test_two_bad_plans_fall_back_to_deterministic():
    """§3: второй провал → фолбэк, не третья попытка. Ролик всё равно соберётся."""
    ws = words()
    llm = client_with(broken_plan(), broken_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 2
    # Фолбэк не ЧИТАЕТ смысл, но иерархию кегля держит: без неё блок выходит
    # стеной ровного крупного текста (С1 сделал кегль иерархией). Ровно одно
    # слово на блок, выбранное детерминированно, — см. `lead_word`.
    for block in edl.captions:
        assert sum(1 for cw in block.words if cw.emphasis.value != "none") == 1


async def test_the_director_says_whose_edit_it_is():
    """Флаг «лист от модели» ставится по принятому листу, а не по факту вызова.

    Раньше джоб смотрел, был ли вызов `direct`. Он бывает и у фолбэка: модель
    дважды ответила негодным листом, расход записан, монтаж детерминированный —
    а в базе стояло «от модели», и статистика качества режиссёра врала.
    """
    ws = words()

    accepted: dict = {}
    await direct(ws, 4.5, llm=client_with(good_plan()), language="ru", cfg=cfg(), report=accepted)
    assert accepted["from_llm"] is True

    fell_back: dict = {}
    llm = client_with(broken_plan(), broken_plan())
    await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg(), report=fell_back)
    assert len(llm.fake.calls) == 2, "модель вызывалась — и всё равно лист не её"
    assert fell_back["from_llm"] is False

    no_llm: dict = {}
    await direct(ws, 4.5, llm=None, language="ru", cfg=cfg(), report=no_llm)
    assert no_llm["from_llm"] is False


async def test_llm_outage_falls_back_without_raising():
    from src.llm.providers import ProviderError

    ws = words()
    llm = LLMClient(cfg=cfg(), provider=FakeProvider(
        [ProviderError("сеть", retryable=True), ProviderError("сеть", retryable=True)]))
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []


async def test_no_llm_at_all_still_produces_edl():
    ws = words()
    edl = await direct(ws, 4.5, llm=None, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == list(range(9))


async def test_llm_returning_garbage_indices_is_survivable():
    """Несуществующие индексы схему не нарушают — их отсеивает сборка EDL."""
    ws = words(6)
    plan = {
        "hook_end_word": 999,
        "drop_words": [-5, 999],
        "blocks": [
            {"words": [0, 1, 2, 999], "register": "poster",
             "accent_word": 999, "accent": "shout", "position": "auto", "follow_face": False},
            {"words": [3, 4, 5], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
        ],
    }
    llm = client_with(plan)
    edl = await direct(ws, 3.0, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == [0, 1, 2, 3, 4, 5]


async def test_out_of_range_value_costs_a_retry():
    """Блок из девяти слов схему нарушает — ловится ДО сборки EDL и лечится ретраем.

    Потолок схемы — потолок тихого регистра (8 слов, §1.4.1): длина блока это
    выбор режиссёра, и до восьми слов она законна. Девять не бывает ни плакатом,
    ни тихим блоком.
    """
    ws = words(9)
    good = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": [0, 1, 2], "register": "poster",
                    "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
                   {"words": [3, 4, 5], "register": "poster",
                    "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
                   {"words": [6, 7, 8], "register": "poster",
                    "accent_word": None, "accent": None, "position": "auto", "follow_face": False}],
    }
    bad = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": list(range(9)), "register": "quiet",
                    "accent_word": None, "accent": None,
                    "position": "top", "follow_face": False}],
    }
    llm = client_with(bad, good)
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 2
    assert "максимум 8" in llm.fake.calls[1]["user"]


async def test_a_long_block_in_the_poster_position_costs_a_retry():
    """Тихий регистр в плакатной позиции — противоречие, и чинит его ретрай.

    Тихий блок измерен только наверху (§1.4.1). Подвинуть его туда самим значило
    бы решать за режиссёра, где висит текст: модель, которая набирает всё
    длинными блоками, молча получила бы ролик, где субтитры уехали наверх
    целиком. Позиция — её решение, и ошибка в нём приезжает ей текстом.
    """
    ws = words(6)
    good = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": [0, 1, 2], "register": "poster",
                    "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
                   {"words": [3, 4, 5], "register": "poster",
                    "accent_word": None, "accent": None, "position": "auto", "follow_face": False}],
    }
    bad = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": list(range(6)), "register": "quiet",
                    "accent_word": None, "accent": None,
                    "position": "auto", "follow_face": False}],
    }
    llm = client_with(bad, good)
    edl = await direct(ws, 3.0, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 2
    assert "тихого блока" in llm.fake.calls[1]["user"]


async def test_a_quiet_block_keeps_its_silence():
    """Тихий блок доезжает до EDL без золота, и `lead_word` в него не лезет.

    С3 поднимает слово в блоке, где режиссёр никого не выделил, — иначе блок
    выходит стеной ровного крупного текста. К тихому регистру это не относится:
    в эталоне у него нет ни одного золотого слова (§1.4.1), а выделение в мелком
    кегле у левого края — не иерархия, а пятно.
    """
    ws = words(10)
    plan = {
        "hook_end_word": 1,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1], "register": "poster",
             "accent_word": 1, "accent": "shout", "position": "auto", "follow_face": False},
            {"words": [2, 3, 4, 5, 6, 7], "register": "quiet",
             "accent_word": None, "accent": None, "position": "top", "follow_face": False},
            {"words": [8, 9], "register": "poster",
             "accent_word": 9, "accent": "accent", "position": "auto", "follow_face": False},
        ],
    }
    edl = await direct(ws, 5.0, llm=client_with(plan), language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    quiet = edl.captions[1]
    assert quiet.position == "top" and len(quiet.words) == 6
    assert all(cw.emphasis.value == "none" for cw in quiet.words)
    # Плакаты рядом иерархию не теряют.
    assert [sum(1 for cw in b.words if cw.emphasis.value != "none") for b in edl.captions] == [1, 0, 1]


async def test_gold_asked_for_a_quiet_block_is_dropped_not_retried():
    """Акцент в тихом блоке — перебор с приёмом, и он снимается молча.

    Так же устроены `anchor_track` и `panel`: лишний приём дешевле снять, чем
    гонять модель по кругу. Регистр при этом модель объявила сама, и позиция у
    него верная — спорить тут не о чем, противоречит только акцент.
    """
    ws = words(8)
    plan = {
        "hook_end_word": None,
        "drop_words": [],
        "blocks": [
            {"words": list(range(6)), "register": "quiet",
             "accent_word": 2, "accent": "shout", "position": "top", "follow_face": False},
            {"words": [6, 7], "register": "poster",
             "accent_word": 7, "accent": "accent", "position": "auto", "follow_face": False},
        ],
    }
    llm = client_with(plan)
    edl = await direct(ws, 4.0, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 1, "ретрая быть не должно"
    assert all(cw.emphasis.value == "none" for cw in edl.captions[0].words)


async def test_dropped_words_are_excluded_from_captions_and_cuts():
    ws = words(9)
    plan = {
        "hook_end_word": 2,
        "drop_words": [4],
        "blocks": [
            {"words": [0, 1, 2], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [3, 5], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [6, 7, 8], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
        ],
    }
    llm = client_with(plan)
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    # Модель просила выбросить «сл4», но выброс — не её решение: режутся только
    # затяжные заминки. Каждое выброшенное слово даёт отдельный рез, а десяток
    # таких резов делал ролик рваным до неразборчивости.
    captioned = [cw.word_idx for b in edl.captions for cw in b.words]
    assert 4 in captioned, "смысловое слово обязано остаться в субтитрах"
    assert not any(c.reason == "drop_filler" for c in edl.cuts)


async def test_a_long_filler_is_still_cut():
    """Затяжную заминку режем — это второе исключение из «не режем»."""
    ws = words(9)
    # Заминка обязана уложиться в свой слот: слово 5 начинается в 2.0.
    ws[4] = Word(idx=4, text="эээ", start=1.6, end=1.98)
    plan = {
        "hook_end_word": 2,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1, 2], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [3, 5], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [6, 7, 8], "register": "poster",
             "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
        ],
    }
    edl = await direct(ws, 4.5, llm=client_with(plan), language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert 4 not in [cw.word_idx for b in edl.captions for cw in b.words]
    assert any(c.reason == "drop_filler" for c in edl.cuts)


async def test_words_table_is_sent_not_raw_text():
    """Модель обязана видеть индексы, иначе ей нечем ссылаться (§0.1)."""
    ws = words(4)
    llm = client_with(
        {
            "hook_end_word": None,
            "drop_words": [],
            "blocks": [{"words": [0, 1, 2, 3], "register": "poster",
                        "accent_word": None, "accent": None,
                        "position": "auto", "follow_face": False}],
        }
    )
    await direct(ws, 2.0, llm=llm, language="ru", cfg=cfg())
    sent = llm.fake.calls[0]["user"]
    assert "   0 сл0" in sent
    assert "   3 сл3" in sent


# ------------------------------------------------------------------ правка распознавания


def heard(*texts: str) -> list[Word]:
    return [Word(idx=i, text=t, start=i * 0.4, end=i * 0.4 + 0.3) for i, t in enumerate(texts)]


async def fix_with(ws: list[Word], fixes: list[dict], **kw):
    provider = FakeProvider([result_of({"fixes": fixes})])
    provider.supports_audio = True
    llm = LLMClient(cfg=cfg(), provider=provider)
    out, n = await clean_transcript(ws, llm=llm, language="ru", cfg=cfg(), **kw)
    return out, n, provider


async def test_the_corrector_hears_the_voice():
    """«Родине» вместо «Зенит» по одному тексту не исправить — только услышав."""
    _, _, provider = await fix_with(heard("фотоаппарат", "Родине"), [], audio=(b"mp3", "audio/mp3"))
    assert provider.calls[0]["audio"] == (b"mp3", "audio/mp3")


async def test_a_glued_brand_comes_back_as_one_word():
    """Whisper слепил «Coffee Boom» в одно слово — в субтитре оно и остаётся одним,
    но с неразрывным пробелом: индексы EDL не сдвигаются, строка не рвётся."""
    out, n, _ = await fix_with(heard("в", "кофейню", "Coffee-bon"), [{"i": 2, "fix": "Coffee Boom"}])
    assert n == 1 and out[2].text == "Coffee Boom"
    assert [w.start for w in out] == [w.start for w in heard("в", "кофейню", "Coffee-bon")]


async def test_a_fix_that_adds_ordinary_words_is_refused():
    """«плёнку, 36» на месте «плёнку» — это вставка слова, а не правка."""
    out, n, _ = await fix_with(heard("всю", "пленку"), [{"i": 1, "fix": "плёнку и кадры"}])
    assert n == 0 and out[1].text == "пленку"


async def test_a_fix_shifted_onto_the_neighbour_is_refused():
    """Модель, потерявшая счёт, пишет в слово текст соседнего — субтитр съехал бы."""
    out, n, _ = await fix_with(
        heard("на", "всю", "пленку", "36"), [{"i": 1, "fix": "плёнку"}, {"i": 2, "fix": "36"}]
    )
    assert n == 0 and [w.text for w in out] == ["на", "всю", "пленку", "36"]
