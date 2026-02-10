# Manual Go-Live Review

Generated: 2026-02-10T07:48:30.211731+00:00

## Explainability Sample (5 random decisions)

| timestamp | ticker | action | source | lint_ok | issues |
| --- | --- | --- | --- | --- | --- |
| 2022-01-11 | VOO | HOLD | fallback | False | action_missing_hu,action_missing_en,model_votes_missing_en,model_votes_missing_hu |
| 2022-01-03 | VOO | HOLD | fallback | False | action_missing_hu,action_missing_en,model_votes_missing_en,model_votes_missing_hu |
| 2022-01-12 | VOO | HOLD | fallback | False | action_missing_hu,action_missing_en,model_votes_missing_en,model_votes_missing_hu |
| 2026-02-09 | VOO | HOLD | fallback | True |  |
| 2022-01-17 | VOO | HOLD | fallback | False | action_missing_hu,action_missing_en,model_votes_missing_en,model_votes_missing_hu |

## Email Usability Sample (5 latest decisions)

### 2026-02-10 VOO
```
Summary:
VOO: Action=HOLD

Details:
VOO: NO_TRADE HOLD
Confidence: 0.00, WF: 0.00, Ensemble: 0.0, Decision quality score: 0.08

Rationale:
- Trading was blocked by policy (reason: FALLBACK_NO_MODELS).
- Original model signal: HOLD.

Model votes:
- 
[Q=0.08 | Conf=0.00 | WF=0.0 | CHAOTIC | Models=0]
```

### 2026-02-09 VOO
```
Summary:
VOO: Action=HOLD

Details:
VOO: NO_TRADE HOLD
Confidence: 0.00, WF: 0.00, Ensemble: 0.0, Decision quality score: 0.08

Rationale:
- Trading was blocked by policy (reason: FALLBACK_NO_MODELS).
- Original model signal: HOLD.

Model votes:
- 
[Q=0.08 | Conf=0.00 | WF=0.0 | CHAOTIC | Models=0]
```

### 2022-01-31 VOO
```
Summary:
VOO: Action=HOLD

Details:
Fallback: no RL models available; no trade executed.
[Q=0.00 | Conf=0.00 | WF=0.0 | CHAOTIC | Models=0]
```

### 2022-01-28 VOO
```
Summary:
VOO: Action=HOLD

Details:
Fallback: no RL models available; no trade executed.
[Q=0.00 | Conf=0.00 | WF=0.0 | CHAOTIC | Models=0]
```

### 2022-01-27 VOO
```
Summary:
VOO: Action=HOLD

Details:
Fallback: no RL models available; no trade executed.
[Q=0.00 | Conf=0.00 | WF=0.0 | CHAOTIC | Models=0]
```
