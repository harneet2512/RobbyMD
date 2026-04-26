# Architecture Diagrams (Mermaid source for rendering)

## 1. Full Pipeline

```mermaid
flowchart TD
    MIC[🎤 Microphone] --> ASR

    subgraph LIVE["Phase 1: Live Encounter"]
        ASR["ASR Pipeline<br/>Whisper large-v3 + WhisperX + pyannote<br/>8 stages, all open-source"] --> ADM["Admission Filter<br/>drop filler, duplicates"]
        ADM --> EXT["Claim Extraction<br/>Opus 4.7 structured output<br/>closed predicate vocab"]
        EXT --> SUP["Supersession Detector<br/>Pass 1: rule-based<br/>Pass 2: e5-small-v2 @ 0.88"]
        SUP --> STORE["Claim Store<br/>SQLite, 10 tables<br/>4 branch projections"]
        STORE --> DIFF["Differential Engine<br/>pure math, zero LLM<br/>81 LR features, 30 citations"]
        DIFF --> VER["Counterfactual Verifier<br/>deterministic selection<br/>+ 1 Opus 4.7 call"]
        VER --> NOTE["SOAP Note Generator<br/>Opus 4.7 + provenance validation"]
        NOTE --> UI["4-Panel UI<br/>React + ReactFlow + Zustand"]
    end

    subgraph POST["Phase 2: Post-Encounter (5 Claude Managed Agents)"]
        STORE --> DOC_A["Doctor Aftercare<br/>queries substrate"]
        STORE --> PAT_A["Patient Aftercare<br/>provenance-backed answers"]
        STORE --> HAND["Shift Handoff<br/>structured handoff doc"]
        STORE --> BIAS["Bias Monitor<br/>anchoring, premature closure"]
        STORE --> COAUTH["Note Co-Author<br/>edits preserving provenance"]
    end

    style LIVE fill:#f0f7ff,stroke:#3b82f6
    style POST fill:#fdf2f8,stroke:#ec4899
```

## 2. Supersession Model

```mermaid
flowchart LR
    C1["claim_001<br/>no known allergies<br/>turn 3, ACTIVE"] -->|"Patient says 'actually,<br/>penicillin' at turn 15"| C1S["claim_001<br/>no known allergies<br/>turn 3, SUPERSEDED"]
    C1S -->|"supersession_edge<br/>PATIENT_CORRECTION<br/>score: 0.91"| C2["claim_047<br/>allergic to penicillin<br/>turn 15, ACTIVE"]
    
    style C1S fill:#fee2e2,stroke:#ef4444
    style C2 fill:#dcfce7,stroke:#22c55e
```

## 3. Provenance Chain

```mermaid
flowchart LR
    S["SOAP sentence"] -->|"[c:claim_id]"| CL["Claim<br/>subject, predicate, value"]
    CL -->|"source_turn_id"| T["Turn<br/>speaker, text, timestamp"]
    T -->|"char_start:char_end"| TX["Original Transcript<br/>exact substring"]
    CL -->|"supersession_edges"| OLD["Old Claim<br/>(if fact changed)"]
    
    style S fill:#fef3c7,stroke:#f59e0b
    style TX fill:#dbeafe,stroke:#3b82f6
```

## 4. Opus 4.7 Integration Map

```mermaid
flowchart TD
    subgraph LIVE["Live Encounter (3 integration points)"]
        O1["Claim Extraction<br/>structured output + few-shot<br/>every conversation turn"]
        O2["Next-Best Question<br/>1 call per ranking shift<br/>≤20 words, clinical phrasing"]
        O3["SOAP Note Generation<br/>with [c:claim_id] markers<br/>provenance-validated"]
    end
    
    subgraph AGENTS["Managed Agents (5 agents, client.beta.agents API)"]
        O4["Doctor Aftercare Agent"]
        O5["Patient Aftercare Agent"]
        O6["Shift Handoff Agent"]
        O7["Diagnostic Bias Monitor"]
        O8["Clinical Note Co-Author"]
    end
    
    OPUS["Claude Opus 4.7"] --> O1
    OPUS --> O2
    OPUS --> O3
    OPUS --> O4
    OPUS --> O5
    OPUS --> O6
    OPUS --> O7
    OPUS --> O8

    style OPUS fill:#7c3aed,stroke:#5b21b6,color:#fff
    style LIVE fill:#f0f7ff,stroke:#3b82f6
    style AGENTS fill:#fdf2f8,stroke:#ec4899
```

## 5. Evaluation Comparison (text for bar chart rendering)

### LongMemEval-S (memory lifecycle)
- Mastra OM: 94.9%
- Mem0: 93.4%
- **RobbyMD: 88.4%**
- EverMemOS: 83.0%
- TiMem: 76.9%
- Zep/Graphiti: 71.2%
- Full-context GPT-4o: 64.0%

### MedXpertQA Text (expert medical reasoning, 10-way MCQ)
- **RobbyMD (Opus 4.7 + RAG): 59.3%**
- GPT-5: ~56%
- **RobbyMD (Opus 4.7 baseline): 55.3%**
- Human expert: ~43%
- DeepSeek-R1: 37.8%
- o3-mini: 37.3%
- GPT-4o: ~30%
- Claude 3.5 Haiku: 17.8%
