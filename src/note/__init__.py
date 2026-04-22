"""SOAP note generation, provenance-validated.

Per `Eng_doc.md §5.5`: the generator queries `list_active_claims` from the
substrate, groups claims by SOAP section via the active pack's
`soap_mapping.json`, and emits a note where every sentence carries
back-links to the claim IDs it derived from.

Public surface:
- `generate_soap_note(conn, *, session_id, dialogue_text, reader, reader_env) -> SOAPResult`
- `SOAPResult` — dataclass with `note_text`, `sentence_provenance`, counts.
"""

from src.note.generator import SOAPResult, generate_soap_note

__all__ = ["SOAPResult", "generate_soap_note"]
