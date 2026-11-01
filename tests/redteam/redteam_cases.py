"""Red-team claim cases for the citation checker (CONTRACTS.md section 7).

Each case is one claim, checked alone against the case store in tests/support/tbxcase.py
(case timezone America/New_York, evening of Mon 2023-03-13, EDT = UTC-4).

- expect="verified": the checker must accept it (no reasons).
- expect="rejected": the checker must reject it with EXACTLY the reason codes in `codes`
  (strict rule: every reason is listed, and nothing else).

`note` names the rule being pinned down. Notes tagged [Cn] rely on a reading of the contract
that docs/SPIKES.md proposes to write into CONTRACTS.md (proposed clarification n).
Cases tagged REAL use model output recorded verbatim in tests/fixtures/spikes/.
"""
from __future__ import annotations

from tbxcase import EVENING, Case, cite, spike_claims

META_CLAIMS = spike_claims("s1_chat_tools_turn2_submit_4k")       # muse-spark-1.3
OPEN_CLAIMS = spike_claims("s4_baseten_chat_tools_turn2_submit")  # muse-spark-open on Baseten

UF, VM, OW, NC, TR, UA = "UNKNOWN_FIELD", "VALUE_MISMATCH", "OUTSIDE_WINDOW", "NO_CITATIONS", "TIME_RANGE_UNSUPPORTED", "UNKNOWN_ARTIFACT"


def UE(t: str) -> str:
    return f"UNGROUNDED_ENTITY:{t}"


def claim(text: str, *citations: dict, time_start: str | None = None, time_end: str | None = None) -> dict:
    return {"text": text, "time_start": time_start, "time_end": time_end, "citations": list(citations)}


ALICE = cite("CALL_ALICE", "contact_name", "Alice")

# ------------------------------------------------------------------------------------------
# MUST-ACCEPT
# ------------------------------------------------------------------------------------------
ACCEPT: list[Case] = [
    Case("A01_real_meta_call_claim", "verified", META_CLAIMS[0],
         note="REAL muse-spark-1.3 output. Phone, ISO timestamp and contact in text all grounded; duration_s '95' vs stored int 95. [C1]"),
    Case("A02_real_meta_location_claim", "verified", META_CLAIMS[1],
         note="REAL muse-spark-1.3 output. lat/lon/accuracy_m copied as strings; ISO timestamp grounded. [C1]"),
    Case("A03_real_open_weights_location_claim", "verified", OPEN_CLAIMS[1],
         note="REAL muse-spark-open output, with time_start/time_end equal to the cited ts. [C1]"),
]

# Phone-format variants must still pass (GAMEPLAN): (cited value, phone as written in the text).
# Stored number is "+1 (555) 010-1234".
PHONE_VARIANTS = [
    ("5550101234", "555-010-1234"),
    ("+15550101234", "+1 555 010 1234"),
    ("1-555-010-1234", "(555) 010-1234"),
    ("(555) 010-1234", "+1.555.010.1234"),
    ("+1 (555) 010-1234", "15550101234"),
]
for i, (cited, written) in enumerate(PHONE_VARIANTS, 1):
    ACCEPT.append(Case(f"A04{chr(96 + i)}_phone_variant_{i}", "verified",
                       claim(f"Alice called {written}.", ALICE, cite("CALL_ALICE", "number", cited)),
                       note="phone rule: digits only, last 10 when both have >=10; phone entity grounded via the same normalization [C7]"))

ACCEPT += [
    Case("A05_short_code_exact_digits", "verified",
         claim("A verification code arrived from short code 72975.",
               cite("MSG_SHORTCODE", "sender_number", "72975"), cite("MSG_SHORTCODE", "body", "verification code is 482913")),
         note="<10 digits compare exactly; 5- and 6-digit runs are not phone entities"),
    Case("A06_ts_local_clock_and_month_day_in_case_tz", "verified",
         claim("At 9:42 PM on March 13, Alice placed an outgoing call.",
               cite("CALL_ALICE", "ts_local", "2023-03-13T21:42:07-04:00"), ALICE, cite("CALL_ALICE", "direction", "outgoing")),
         note="ts_local compared after conversion to UTC; clock_time and date read in the case tz (UTC date is Mar 14)"),
    Case("A07_timestamp_minute_precision", "verified",
         claim("Alice placed a call.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42Z")),
         note="cited value without seconds compares at minute precision (stored 01:42:07)"),
    Case("A08_timestamp_without_offset_is_case_local", "verified",
         claim("Alice placed a call.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-13T21:42:07")),
         note="a timestamp value with no offset is read in the case timezone"),
    Case("A09_body_quote_folded", "verified",
         claim("Alice asked to meet at the Riverside garage.",
               cite("MSG_ALICE", "sender", "Alice"), cite("MSG_ALICE", "body", "riverside   GARAGE at 9:45")),
         note="text rule: NFKC + casefold + whitespace collapse, then substring >= 8 chars"),
    Case("A10_body_quote_nfkc_exactly_8_chars", "verified",
         claim("Alice mentioned a blue bag.",
               cite("MSG_ALICE", "sender", "Alice"), cite("MSG_ALICE", "body", "ｂｌｕｅ ｂａｇ")),
         note="fullwidth 'blue bag' folds to 8 chars: the >=8 boundary is inclusive and measured after folding"),
    Case("A11_coordinates_within_1e-4", "verified",
         claim("At 2023-03-14T02:50:00Z the phone was near 40.7061, -74.0087.",
               cite("LOC_PRECISE", "ts_utc", "2023-03-14T02:50:00Z"),
               cite("LOC_PRECISE", "lat", "40.7061"), cite("LOC_PRECISE", "lon", "-74.0087")),
         note="stored 40.706123/-74.008701; values and the coordinate entity within 1e-4 [C1]"),
    Case("A12_score_within_1e-3", "verified",
         claim("SAM tagged a car in a photo.",
               cite("TAG_CAR", "label", "car"), cite("TAG_CAR", "score", "0.973")),
         note="score rule: abs diff <= 1e-3 (stored 0.9726...)"),
    Case("A13_contact_grounded_by_transcript_quote", "verified",
         claim("In a voice note, Alice asked to meet at the Riverside parking garage.",
               cite("TRN_VOICE", "text", "it's Alice, meet me at the Riverside parking garage")),
         note="real Voice Transcribe text; contact grounded because it appears inside a cited value [C8]"),
    Case("A14_call_range_includes_duration", "verified",
         claim("Alice's call lasted 95 seconds.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z"),
               cite("CALL_ALICE", "duration_s", "95"),
               time_start="2023-03-14T01:42:07Z", time_end="2023-03-14T01:43:42Z"),
         note="R5: for calls the latest support point is ts_utc + duration_s"),
    Case("A15_time_range_edges_exactly_60s", "verified",
         claim("Alice placed a call.", ALICE, cite("CALL_ALICE", "duration_s", "95"),
               time_start="2023-03-14T01:41:07Z", time_end="2023-03-14T01:44:42Z"),
         note="R5 slack is inclusive: exactly 60 s before ts and 60 s after ts+duration [C5]"),
    Case("A16_window_end_inclusive", "verified",
         claim("Alice placed a call.", ALICE), window=("2023-03-14T01:00:00Z", "2023-03-14T01:42:07Z"),
         note="Window ends are inclusive (CONTRACTS section 3)"),
    Case("A17_null_timestamp_ok_without_window", "verified",
         claim("A draft message said to delete everything before Friday.",
               cite("MSG_DRAFT", "body", "delete everything before Friday")), window=None,
         note="R3 only applies when a window is set"),
    Case("A18_cross_artifact_call_and_location", "verified",
         claim("At 9:42 PM Alice called while the phone was at 40.7484, -73.9857.",
               cite("CALL_ALICE", "ts_local", "2023-03-13T21:42:07-04:00"), ALICE,
               cite("LOC_SPIKE", "lat", "40.7484"), cite("LOC_SPIKE", "lon", "-73.9857")),
         note="one claim citing two artifacts; clock_time matches one cited timestamp; coordinate pair grounded"),
    Case("A19_iso_date_in_case_tz", "verified",
         claim("On 2023-03-13 (local time) Alice placed an outgoing call.",
               cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z"), ALICE, cite("CALL_ALICE", "direction", "outgoing")),
         note="date entity is matched against the cited timestamp's date in the case tz, not the UTC date; a date is not a phone [C1]"),
    Case("A20_url_and_email_grounded", "verified",
         claim("Carol asked for the photos to be uploaded to https://files.example.net/x7 and emailed to ops@example.org.",
               cite("MSG_CAROL", "sender", "Carol"),
               cite("MSG_CAROL", "body", "Upload the photos to https://files.example.net/x7 and email ops@example.org")),
         window=None, note="url and email follow the general R4 rule: present in a cited value [C8]"),
    Case("A21_casefold_exact_match_fields", "verified",
         claim("Alice placed an outgoing call.",
               cite("CALL_ALICE", "contact_name", "ALICE"), cite("CALL_ALICE", "direction", "Outgoing")),
         note="'everything else' rule: NFKC + casefold + whitespace collapse, then exact"),
]

# ------------------------------------------------------------------------------------------
# MUST-REJECT
# ------------------------------------------------------------------------------------------
REJECT: list[Case] = [
    Case("R01_fabricated_artifact_id", "rejected",
         claim("A call was placed.", cite("call_0000000000", "direction", "outgoing")), window=None,
         codes=frozenset({UA}), indexes={UA: {0}}),
    Case("R02_tool_call_id_cited_as_artifact", "rejected",
         claim("An outgoing call was placed.", cite("call_0d96ca43-8907-4d91-aac2-346156bd69d6", "direction", "outgoing")),
         codes=frozenset({UA}), indexes={UA: {0}},
         note="REAL: Meta returns tool_call ids shaped call_<uuid>, easy to confuse with call_ artifact ids. "
              "An unknown artifact gets only UNKNOWN_ARTIFACT, not R2/R3 as well [C2]"),
    Case("R03_raw_json_not_citable", "rejected",
         claim("Alice placed a call.", cite("CALL_ALICE", "raw_json", "{}"), ALICE),
         codes=frozenset({UF}), indexes={UF: {0}}),
    Case("R04_tz_basis_and_ts_original_not_citable", "rejected",
         claim("A call was placed.", cite("CALL_ALICE", "tz_basis", "utc_epoch_ms"), cite("CALL_ALICE", "ts_original", "1678758127000")),
         codes=frozenset({UF}), indexes={UF: {0, 1}}),
    Case("R05_field_from_another_table", "rejected",
         claim("A call was placed.", cite("CALL_ALICE", "body", "Meet me")),
         codes=frozenset({UF}), indexes={UF: {0}}),
    Case("R06_wrong_contact_name", "rejected",
         claim("Alicia was called.", cite("CALL_ALICE", "contact_name", "Alicia")),
         codes=frozenset({VM}), indexes={VM: {0}}, note="'Alicia' is not the known contact 'Alice' (whole-word match)"),
    Case("R07_phone_one_digit_off", "rejected",
         claim("A call was placed.", cite("CALL_ALICE", "number", "+1 (555) 010-1235")),
         codes=frozenset({VM}), indexes={VM: {0}}),
    Case("R08_short_code_suffix_is_not_a_match", "rejected",
         claim("A code arrived from a short code.", cite("MSG_SHORTCODE", "sender_number", "2975")),
         codes=frozenset({VM}), indexes={VM: {0}}, note="<10 digits on either side compare ALL digits exactly"),
    Case("R09_quote_7_chars_too_short", "rejected",
         claim("A message mentioned a place.", cite("MSG_ALICE", "body", "Riversi")),
         codes=frozenset({VM}), indexes={VM: {0}}, note="substring of the body but only 7 chars (twin of A10)"),
    Case("R10_timestamp_one_minute_off", "rejected",
         claim("A call was placed.", cite("CALL_ALICE", "ts_utc", "2023-03-14T01:43:07Z")),
         codes=frozenset({VM}), indexes={VM: {0}}),
    Case("R11_ts_local_with_pre_dst_offset", "rejected",
         claim("A call was placed.", cite("CALL_ALICE", "ts_local", "2023-03-13T21:42:07-05:00")),
         codes=frozenset({VM}), indexes={VM: {0}}, note="EST offset after the Mar 12 DST change is 02:42:07Z, a different instant"),
    Case("R12_latitude_off_by_2e-4", "rejected",
         claim("The phone's location was recorded.", cite("LOC_SPIKE", "lat", "40.7486")),
         codes=frozenset({VM}), indexes={VM: {0}}),
    Case("R13_score_off_by_more_than_1e-3", "rejected",
         claim("A photo was tagged.", cite("TAG_CAR", "score", "0.975")),
         codes=frozenset({VM}), indexes={VM: {0}}),
    Case("R14_wrong_direction", "rejected",
         claim("A call was received.", cite("CALL_ALICE", "direction", "incoming")),
         codes=frozenset({VM}), indexes={VM: {0}}),
    Case("R15_artifact_outside_window", "rejected",
         claim("A message was received.", cite("MSG_CAROL", "sender", "Carol")),
         codes=frozenset({OW}), indexes={OW: {0}}, note="08:00 local is outside the 21:00-23:00 window"),
    Case("R16_null_timestamp_with_window", "rejected",
         claim("A draft message said to delete everything.", cite("MSG_DRAFT", "body", "delete everything before Friday")),
         codes=frozenset({OW}), indexes={OW: {0}}, detail_contains={OW: "no timestamp"}),
    Case("R17_one_second_before_window_start", "rejected",
         claim("A call was placed.", ALICE), window=("2023-03-14T01:42:08Z", "2023-03-14T03:00:00Z"),
         codes=frozenset({OW}), indexes={OW: {0}}, note="twin of A16"),
    Case("R18_ungrounded_phone", "rejected",
         claim("Alice called +1 555 999 0000.", ALICE, cite("CALL_ALICE", "number", "+1 (555) 010-1234")),
         codes=frozenset({UE("phone")})),
    Case("R19_ungrounded_contact", "rejected",
         claim("Alice and Bob were on the call.", ALICE),
         codes=frozenset({UE("contact")}), note="Bob is a known contact but nothing cited mentions Bob"),
    Case("R20_wrong_clock_time", "rejected",
         claim("Alice called at 9:15 PM.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z")),
         codes=frozenset({UE("clock_time")})),
    Case("R21_utc_clock_time_in_text", "rejected",
         claim("Alice called at 01:42.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z")),
         codes=frozenset({UE("clock_time")}), note="01:42 is the UTC clock; in the case tz the call was at 21:42"),
    Case("R22_utc_date_in_text", "rejected",
         claim("On 2023-03-14 Alice placed a call.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z")),
         codes=frozenset({UE("date")}), note="UTC date; the case-local date is 2023-03-13 (twin of A19); a date is not a phone [C1]"),
    Case("R23_wrong_month_day", "rejected",
         claim("On Mar 14, Alice placed a call.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z")),
         codes=frozenset({UE("date")})),
    Case("R24_ungrounded_coordinates", "rejected",
         claim("The phone was at 40.7580, -73.9855.", cite("LOC_SPIKE", "lat", "40.7484"), cite("LOC_SPIKE", "lon", "-73.9857")),
         codes=frozenset({UE("coordinate")})),
    Case("R25_ungrounded_email", "rejected",
         claim("Carol asked for the files to be emailed to boss@example.org.",
               cite("MSG_CAROL", "sender", "Carol"), cite("MSG_CAROL", "body", "email ops@example.org")),
         window=None, codes=frozenset({UE("email")})),
    Case("R26_ungrounded_url", "rejected",
         claim("Carol shared https://evil.example.com/drop.",
               cite("MSG_CAROL", "sender", "Carol"), cite("MSG_CAROL", "body", "https://files.example.net/x7")),
         window=None, codes=frozenset({UE("url")})),
    Case("R27_real_open_weights_restates_window", "rejected", OPEN_CLAIMS[0],
         codes=frozenset({UE("timestamp")}),
         note="REAL muse-spark-open output: the text restates the question window "
              "(2023-03-14T01:00:00Z / 03:00:00Z), which no citation supports [C1]"),
    Case("R28_time_start_after_time_end", "rejected",
         claim("A call was placed.", cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z"),
               time_start="2023-03-14T01:43:00Z", time_end="2023-03-14T01:42:00Z"),
         codes=frozenset({TR})),
    Case("R29_time_end_61s_past_call_end", "rejected",
         claim("Alice placed a call.", ALICE, cite("CALL_ALICE", "duration_s", "95"),
               time_start="2023-03-14T01:42:07Z", time_end="2023-03-14T01:44:43Z"),
         codes=frozenset({TR}), note="twin of A15"),
    Case("R30_range_stretched_to_7pm", "rejected",
         claim("Alice was on the phone during the evening.", ALICE, cite("CALL_ALICE", "ts_utc", "2023-03-14T01:42:07Z"),
               time_start="2023-03-13T19:00:00-04:00", time_end="2023-03-13T21:43:00-04:00"),
         codes=frozenset({TR})),
    Case("R31_no_citations", "rejected",
         claim("The phone stayed at home all evening."),
         codes=frozenset({NC})),
    Case("R32_no_citations_lists_every_reason", "rejected",
         claim("Alice called at 9:42 PM."),
         codes=frozenset({NC, UE("contact"), UE("clock_time")}), note="strict rule: every reason is listed"),
    Case("R33_one_bad_citation_rejects_whole_claim", "rejected",
         claim("Alice placed a call.", ALICE, cite("call_ffffffffff", "direction", "outgoing")), window=None,
         codes=frozenset({UA}), indexes={UA: {1}}, note="strict rule (g)"),
    Case("R34_three_reasons_with_indexes", "rejected",
         claim("A message and a call were recorded.",
               cite("CALL_ALICE", "raw_json", "{}"), cite("MSG_CAROL", "sender", "Carol"), cite("CALL_ALICE", "direction", "incoming")),
         codes=frozenset({UF, OW, VM}), indexes={UF: {0}, OW: {1}, VM: {2}}),
    Case("R35_hallucinated_transcript_quote", "rejected",
         claim("In a voice note someone asked to meet at the Riverside parking garage.",
               cite("TRN_VOICE", "text", "meet me at the Riverside parking garage at 10 tonight")),
         codes=frozenset({VM}), indexes={VM: {0}}),
    Case("R36_wrong_media_path", "rejected",
         claim("A photo was taken.", cite("MED_PHOTO", "file_path", "DCIM/Camera/IMG_20230313_214501.jpg")),
         codes=frozenset({VM}), indexes={VM: {0}}),
]

CASES: list[Case] = ACCEPT + REJECT
assert len({c.id for c in CASES}) == len(CASES), "duplicate case ids"
assert all(c.expect == "verified" for c in ACCEPT) and all(c.expect == "rejected" and c.codes for c in REJECT)
