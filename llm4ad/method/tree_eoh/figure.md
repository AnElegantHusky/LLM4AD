graph TB
    subgraph "Population Management (TreePopulation)"
        direction TB
        subgraph "Logical Tree Structure"
            N0_0(Root Node L0) --> N1_1(Node L1)
            N0_0 --> N1_2(Node L1)
            N1_1 --> N2_1(Leaf Node L2)
            N1_1 --> N2_2(Leaf Node L2)
            N1_2 --> N2_3(Leaf Node L2)
            style N2_1 fill:#d4edda,stroke:#28a745
            style N2_2 fill:#d4edda,stroke:#28a745
            style N2_3 fill:#d4edda,stroke:#28a745
        end

        subgraph "Synchronized Indices (O(1) Access)"
            IdxID[ID Index: Dict{ID: Node}]
            IdxLevel[Level Index: Dict{Level: List[Node]}]
            IdxLeaf[Leaf Index: Set{Node}]
        end

        %% Connections between indices and tree nodes (Conceptual)
        IdxID -.-> N0_0 & N1_1 & N2_1
        IdxLevel -.-> N0_0 & N1_1 & N2_1
        IdxLeaf -.-> N2_1 & N2_2 & N2_3

        TabuList["Tabu Dictionary (Per Prompt Type)"]
    end

    subgraph "Selection Mechanism (TreePopulation.select)"
        direction TB
        InputNodes[Candidate Nodes from Indices] --> FilterTabu{Exclude Tabu IDs?}
        FilterTabu -- Yes --> Stage1

        subgraph "Hierarchical Selection Strategy"
            Stage1{"Stage 1: Priority on Exploration\n(Non-Tabu Leaves with Min Level)"}
            Stage1 -- "Not Enough" --> Stage2{"Stage 2: Shallow Node Fallback\n(Non-Tabu Nodes with Min Level)"}
            Stage2 -- "Not Enough" --> Stage3{"Stage 3: Exploitation Fallback\n(Probabilistic based on Score)"}
        end
        Stage1 -- "Selected" --> SelectedParents
        Stage2 -- "Selected" --> SelectedParents
        Stage3 -- "Selected" --> SelectedParents

        SelectedParents[Selected Parents]
    end

    subgraph "Evolutionary Loop (TreeEoH)"
        direction LR
        Gen[Generation\n(LLM + Prompts E1/E2/M1/M2)]
        Eval[Evaluation\n(Thread/Process Pool)]
        CheckDup{Is Duplicate ID?}
        Register[Register Node]
        Feedback[Feedback Mechanism]

        SelectedParents --> Gen
        Gen --> Eval
        Eval --> CheckDup
        CheckDup -- "No (Unique)" --> Register
        CheckDup -- "Yes (Duplicate)" --> Feedback
    end

    %% Main Flow Connections
    Register ==>|"Add Node & Update All Indices"| IdxID
    Register ==>|"Update Leaf Status"| IdxLeaf
    Feedback -.->|"Mark Parents as Tabu"| TabuList
    TabuList -.-> FilterTabu

    style IdxID fill:#e2e3e5,stroke:#6c757d
    style IdxLevel fill:#e2e3e5,stroke:#6c757d
    style IdxLeaf fill:#e2e3e5,stroke:#6c757d
    style SelectedParents fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    style Register fill:#cce5ff,stroke:#007bff,stroke-width:2px
    style Feedback fill:#f8d7da,stroke:#dc3545