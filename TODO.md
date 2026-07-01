# TODO - Criteria PASS/FAIL/REVIEW enforcement

- [x] Update `Evalution/verification_engine/numeric_verifier.py`
  - [x] COUNT emits `UNVERIFIED` instead of informational-only
  - [x] EXISTENCE emits `UNVERIFIED` instead of informational-only

- [x] Update `Evalution/verification_engine/engine.py`
  - [x] PASS blocks when COUNT/EXISTENCE facts are `UNVERIFIED`
  - [x] FAIL remains restricted to numeric THRESHOLD/COMPARISON provable FALSE

- [ ] Fix general wrong PASS for “average annual turnover over past N financial years”
  - [ ] Update `Evalution/verification_engine/criterion_parser.py` to preserve “past N financial years” context in the generated `RequirementCheck.description`
  - [ ] Update `Evalution/verification_engine/numeric_verifier.py` `_verify_threshold()` to:
    - [ ] detect “past N financial years” from `requirement.description`/context
    - [ ] only compute averages when evidence can clearly map turnover numbers to distinct financial years
    - [ ] otherwise emit `UNVERIFIED` (=> REVIEW), not PASS
