import csv

from backend.ir_system.config import settings
from backend.ir_system.service import SearchService


def test_ocr_sidecars_make_every_document_searchable() -> None:
    service = SearchService()
    service.ensure_ready()

    assert service.summary is not None
    assert service.summary.document_count == 52
    assert service.summary.page_count == 171
    assert len(service.summary.ocr_sidecar_documents) == 2
    assert service.summary.scanned_or_empty_documents == []


def test_national_and_international_awards_are_not_confused() -> None:
    service = SearchService()

    national = service.query(
        "What is the maximum limit of the grant for the National Travel Award?",
        answer_mode="extractive",
    )
    international = service.query(
        "What is the maximum limit of the grant for the International Travel Award?",
        answer_mode="extractive",
    )

    assert "₹25,000" in national.answer.answer
    assert "INR 1.5 lakh" in international.answer.answer
    assert national.results[0].chunk.file_name == "Travel-Grant-for-Registered-Ph.d-Student.pdf"
    assert international.results[0].chunk.file_name == "BITS-Pilani-International-Travel-Award_Guidelines-1.pdf"


def test_table_conditions_produce_the_correct_overhead_allocations() -> None:
    service = SearchService()
    common = (
        "for Professional Development Fund (PDF), Department Development Fund (DDF), "
        "and Campus Development Fund (CDF) when overhead is "
    )

    lower = service.query("What are the allocation percentages " + common + "less than 10%?")
    upper = service.query("What are the allocation percentages " + common + "more than 10%?")
    generic = service.query("How is overhead cost distributed among different funds?")

    assert "PDF): 40%" in lower.answer.answer
    assert "DDF): 20%" in lower.answer.answer
    assert "CDF): 40%" in lower.answer.answer
    assert "PDF): 60%" in upper.answer.answer
    assert "DDF): 20%" in upper.answer.answer
    assert "CDF): 20%" in upper.answer.answer
    assert not generic.answer.abstained
    assert "Professional Development Fund" in generic.answer.answer


def test_interleaved_department_table_returns_the_complete_cs_is_list() -> None:
    result = SearchService().query(
        "What are the proposed sub-areas for PhD Qualifying examination for the CS&IS department?"
    )

    assert not result.answer.abstained
    assert "AI, Machine Learning & Data Mining" in result.answer.answer
    assert "Networking & Mobile Computing" in result.answer.answer
    assert "Algorithms Theoretical Computer Science" in result.answer.answer


def test_out_of_scope_question_abstains() -> None:
    result = SearchService().query("Who is the captain of the institute football team?")

    assert result.answer.abstained
    assert result.answer.confidence == "low"


def test_out_of_scope_paraphrases_do_not_get_plausible_but_unrelated_answers() -> None:
    questions = [
        "latest PhD application date?",
        "PhD tuition fee for 2026?",
        "Who won the best research award this year?",
        "Can PhD students bring pets into the hostel?",
        "Can PhD students bring pets to campus?",
        "How much does campus accommodation cost for research scholars?",
        "iit bombay phd fellowship amount?",
        "Stanford University PhD fellowship amount?",
        "up-to-date PhD application closing date?",
        "upcoming PhD application deadline?",
        "current PhD stipend?",
        "present PhD tuition fee?",
        "newest PhD application date?",
        "What is the PhD stipend this academic year?",
        "PhD application deadline this admission cycle?",
        "PhD application deadline in 2015?",
        "Who is this year's best research scholar?",
        "PhD tuition fee in 2011?",
        "What is the CSIR NET stipend?",
        "nasa research budget?",
        "Cambridge PhD fellowship amount?",
        "University of Oxford PhD fellowship amount?",
        "University of Hyderabad PhD fellowship amount?",
        "Hyderabad University PhD fellowship amount?",
        "Goa University PhD fellowship amount?",
        "how much fellowship does oxford pay phd researchers?",
        "How much is campus accommodation for research scholars?",
        "Are research scholars allowed to bring puppies or birds to campus?",
        "What are the hostel charges for PhD students?",
    ]
    service = SearchService()

    assert all(service.query(question).answer.abstained for question in questions)


def test_supported_policy_acronyms_are_not_mistaken_for_external_entities() -> None:
    questions = [
        "Are PhD students at the KK Birla Goa Campus eligible for the International Travel Award?",
        "What ID should a PhD student provide on the travel grant form?",
        "What CGPA is required for the PhD fellowship?",
    ]
    service = SearchService()

    assert all(not service.query(question).answer.abstained for question in questions)


def test_ta_da_queries_reach_the_reviewed_policy_text() -> None:
    service = SearchService()

    rules = service.query("What are the TA/DA reimbursement rules?")
    deadline = service.query("When must the TA/DA form be submitted?")
    dotted = service.query("When must the T.A./D.A. form be filed?")
    hyphenated = service.query("When is the TA-DA claim due?")
    ampersand = service.query("When must the T.A. & D.A. form be filed?")
    word_connector = service.query("When must the T.A. and D.A. form be filed?")
    dot_connector = service.query("When must the TA.DA form be filed?")
    fully_dotted = service.query("When must the T.A.D.A. form be filed?")

    assert not rules.answer.abstained
    assert not deadline.answer.abstained
    assert not dotted.answer.abstained
    assert not hyphenated.answer.abstained
    assert not ampersand.answer.abstained
    assert not word_connector.answer.abstained
    assert not dot_connector.answer.abstained
    assert not fully_dotted.answer.abstained
    assert rules.results[0].chunk.file_name == "Travel-Grant-for-Registered-Ph.d-Student.pdf"
    assert deadline.results[0].chunk.file_name == "Travel-Grant-for-Registered-Ph.d-Student.pdf"
    assert "within 15 days" in dotted.answer.answer
    assert "within 15 days" in hyphenated.answer.answer
    assert "within 15 days" in ampersand.answer.answer
    assert "within 15 days" in word_connector.answer.answer
    assert "within 15 days" in dot_connector.answer.answer
    assert "within 15 days" in fully_dotted.answer.answer


def test_identifier_scope_does_not_depend_on_requested_result_count() -> None:
    service = SearchService()

    for top_k in (1, 3, 5):
        result = service.query(
            "What is the role of ARD in a PhD transfer?",
            top_k=top_k,
            answer_mode="extractive",
        )
        assert not any("outside this corpus" in warning for warning in result.warnings)
        assert result.answer.abstained


def test_rare_identifiers_ground_ranking_and_answer_selection() -> None:
    service = SearchService()

    fcra = service.query("What does FCRA govern in sponsored research?")
    qe = service.query("What does QE mean in the PhD programme?")
    unsupported_amount = service.query("UGC NET fellowship amount?")
    undefined_id = service.query("What does ID mean on the travel grant form?")
    undefined_ard = service.query("What does ARD stand for in PhD records?")
    undefined_net = service.query("What does NET stand for in the PhD fellowship policy?")
    undefined_ugc = service.query("What does UGC mean in the PhD fellowship policy?")
    spelled_out_id = service.query(
        "What identification number should be entered on the travel grant form?"
    )

    assert not fcra.answer.abstained
    assert "FCRA account" in fcra.answer.answer
    assert not qe.answer.abstained
    assert "Qualifying Examination" in qe.answer.answer
    assert unsupported_amount.answer.abstained
    assert undefined_id.answer.abstained
    assert undefined_ard.answer.abstained
    assert undefined_net.answer.abstained
    assert undefined_ugc.answer.abstained
    assert not spelled_out_id.answer.abstained
    assert "ID" in spelled_out_id.answer.answer


def test_travel_lodging_reimbursement_is_not_confused_with_hostel_charges() -> None:
    result = SearchService().query(
        "What accommodation cost or rate is reimbursed under the National Travel Grant?"
    )

    assert not result.answer.abstained
    assert "₹1,500" in result.answer.answer


def test_missing_contingency_item_policy_is_reported_honestly() -> None:
    result = SearchService().query(
        "What are the items which can be purchased under the contingency with Institute PhD fellowship?"
    )

    assert result.results[0].chunk.file_name == "Institute_Contigency-Form.pdf"
    assert result.answer.abstained
    assert "does not enumerate" in result.answer.answer
    assert result.warnings


def test_missing_publication_count_rule_is_reported_honestly() -> None:
    result = SearchService().query(
        "What quality and number of publications are required for my PhD thesis work?"
    )

    assert "CheckList_PhD-Thesis-submission.pdf" in {
        item.chunk.file_name for item in result.results[:3]
    }
    assert result.answer.abstained
    assert "does not state a minimum" in result.answer.answer


def test_missing_general_admission_criteria_are_not_confused_with_grant_criteria() -> None:
    result = SearchService().query(
        "What are the eligibility criteria for admission in the full-time PhD programme?"
    )

    assert result.results[0].chunk.file_name == "DRC_Guidelines-2015-updated.pdf"
    assert result.answer.abstained
    assert "different question" in result.answer.answer


def test_missing_thesis_language_rule_is_reported_as_a_source_gap() -> None:
    result = SearchService().query("In what language should the PhD thesis be written?")

    assert result.answer.abstained
    assert "do not state" in result.answer.answer


def test_all_labeled_unanswerable_queries_abstain() -> None:
    with settings.unanswerable_dataset.open(encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    service = SearchService()

    failures = [
        row["query_id"]
        for row in cases
        if not service.query(row["Question"]).answer.abstained
    ]

    assert failures == []
