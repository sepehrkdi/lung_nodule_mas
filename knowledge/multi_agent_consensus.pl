/**
 * MULTI-AGENT CONSENSUS KNOWLEDGE BASE
 * =====================================
 * 
 * EDUCATIONAL PURPOSE - SYMBOLIC AI FOR MULTI-AGENT SYSTEMS:
 * 
 * This Prolog knowledge base extends the Lung-RADS rules to support:
 * 1. Multi-agent evidence fusion from 3 radiologists + 3 pathologists
 * 2. Weighted voting with agent-specific reliability scores
 * 3. Disagreement detection and resolution strategies
 * 4. TNM staging from publicly available criteria
 * 
 * AGENT TYPES SUPPORTED:
 * - radiologist_densenet: DenseNet121 CNN (weight: 1.0)
 * - radiologist_resnet:   ResNet50 CNN (weight: 1.0)
 * - radiologist_rules:    Size-based heuristic (weight: 0.7)
 * - pathologist_regex:    Regex-based extraction (weight: 0.8)
 * - pathologist_spacy:    spaCy NER + rules (weight: 0.9)
 * 
 * REFERENCES:
 * - ACR Lung-RADS v1.1 (2019): https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads
 * - AJCC TNM Staging 8th Ed: https://www.cancer.org/cancer/lung-cancer/detection-diagnosis-staging/staging.html
 */


/* ============================================================
 * SECTION 1: AGENT REGISTRY AND WEIGHTS
 * ============================================================
 * Define agent types and their reliability weights for voting.
 */

% Agent type definitions
agent_type(radiologist_densenet, radiologist, cnn).
agent_type(radiologist_resnet, radiologist, cnn).
agent_type(radiologist_rules, radiologist, rule_based).
agent_type(pathologist_regex, pathologist, regex).
agent_type(pathologist_spacy, pathologist, nlp).

% Agent weights for consensus voting
% Higher weight = more influence on final decision
% NOTE: These are DEFAULT/FALLBACK weights.
% At runtime, DynamicWeightCalculator (models/dynamic_weights.py)
% computes per-case weights based on information richness and
% asserts them into the KB, overriding these defaults.
:- dynamic agent_weight/2.
agent_weight(radiologist_densenet, 1.0).
agent_weight(radiologist_resnet, 1.0).
agent_weight(radiologist_rules, 0.7).
agent_weight(pathologist_regex, 0.8).
agent_weight(pathologist_spacy, 0.9).

% Get weight for an agent (with default)
get_agent_weight(Agent, Weight) :-
    agent_weight(Agent, Weight), !.
get_agent_weight(_, 0.5).  % Default weight

% Check if agent is a radiologist
is_radiologist(Agent) :-
    agent_type(Agent, radiologist, _).

% Check if agent is a pathologist
is_pathologist(Agent) :-
    agent_type(Agent, pathologist, _).


/* ============================================================
 * SECTION 2: DYNAMIC PREDICATES FOR AGENT FINDINGS
 * ============================================================
 * These are asserted by Python agents during runtime.
 */

:- dynamic agent_finding/4.         % agent_finding(NoduleId, Agent, Probability, Class)
:- dynamic agent_features/3.        % agent_features(NoduleId, Agent, Features)
:- dynamic agent_confidence/3.      % agent_confidence(NoduleId, Agent, Confidence)
:- dynamic nlp_entity/4.            % nlp_entity(NoduleId, Agent, EntityType, Value)
:- dynamic nodule_size/2.           % nodule_size(NoduleId, SizeMM)
:- dynamic nodule_texture/2.        % nodule_texture(NoduleId, TextureType)
:- dynamic consensus_result/4.      % consensus_result(NoduleId, Probability, Class, Confidence)


/* ============================================================
 * SECTION 3: LUNG-RADS v1.1 CLASSIFICATION (ACR 2019)
 * ============================================================
 * Official Lung-RADS categories based on nodule characteristics.
 * Source: American College of Radiology Lung-RADS v1.1
 */

% Lung-RADS Category 0: Incomplete
lung_rads(_, 0, 'Incomplete - prior CT required') :-
    fail.  % Not implemented - requires prior comparison

% Lung-RADS Category 1: Negative (no nodules)
lung_rads(NoduleId, 1, 'Negative - no nodules') :-
    \+ nodule_size(NoduleId, _).

% Lung-RADS Category 2: Benign Appearance
% Solid nodule <6mm
lung_rads(NoduleId, 2, 'Benign - solid <6mm') :-
    nodule_size(NoduleId, Size),
    Size < 6,
    nodule_texture(NoduleId, solid).

% Part-solid nodule <6mm
lung_rads(NoduleId, 2, 'Benign - part-solid <6mm') :-
    nodule_size(NoduleId, Size),
    Size < 6,
    nodule_texture(NoduleId, part_solid).

% Ground-glass nodule <30mm
lung_rads(NoduleId, 2, 'Benign - GGN <30mm') :-
    nodule_size(NoduleId, Size),
    Size < 30,
    nodule_texture(NoduleId, ground_glass).

% Perifissural nodule <10mm
lung_rads(NoduleId, 2, 'Benign - perifissural') :-
    nodule_size(NoduleId, Size),
    Size < 10,
    nlp_entity(NoduleId, _, location, perifissural).

% Lung-RADS Category 3: Probably Benign
% Solid nodule 6-<8mm
lung_rads(NoduleId, 3, 'Probably benign - solid 6-8mm') :-
    nodule_size(NoduleId, Size),
    Size >= 6, Size < 8,
    nodule_texture(NoduleId, solid).

% Part-solid nodule 6-<8mm total
lung_rads(NoduleId, 3, 'Probably benign - part-solid 6-8mm') :-
    nodule_size(NoduleId, Size),
    Size >= 6, Size < 8,
    nodule_texture(NoduleId, part_solid).

% GGN >=30mm
lung_rads(NoduleId, 3, 'Probably benign - GGN >=30mm') :-
    nodule_size(NoduleId, Size),
    Size >= 30,
    nodule_texture(NoduleId, ground_glass).

% Lung-RADS Category 4A: Suspicious
% Solid nodule 8-<15mm
lung_rads(NoduleId, '4A', 'Suspicious - solid 8-15mm') :-
    nodule_size(NoduleId, Size),
    Size >= 8, Size < 15,
    nodule_texture(NoduleId, solid).

% Part-solid with solid component >=6mm
lung_rads(NoduleId, '4A', 'Suspicious - part-solid >=6mm solid') :-
    nodule_size(NoduleId, Size),
    Size >= 6,
    nodule_texture(NoduleId, part_solid).

% Lung-RADS Category 4B: Very Suspicious
% Solid nodule >=15mm
lung_rads(NoduleId, '4B', 'Very suspicious - solid >=15mm') :-
    nodule_size(NoduleId, Size),
    Size >= 15,
    nodule_texture(NoduleId, solid).

% Part-solid with solid component >=8mm
lung_rads(NoduleId, '4B', 'Very suspicious - part-solid >=8mm solid') :-
    nodule_size(NoduleId, Size),
    Size >= 8,
    nodule_texture(NoduleId, part_solid),
    nlp_entity(NoduleId, _, solid_component, SolidSize),
    SolidSize >= 8.

% Lung-RADS Category 4X: Additional suspicious features
lung_rads(NoduleId, '4X', 'Very suspicious - additional features') :-
    lung_rads(NoduleId, Cat, _),
    (Cat == '4A' ; Cat == '4B'),
    has_suspicious_feature(NoduleId).

% Suspicious features that upgrade to 4X
has_suspicious_feature(NoduleId) :-
    nlp_entity(NoduleId, _, spiculation, marked).
has_suspicious_feature(NoduleId) :-
    nlp_entity(NoduleId, _, lymphadenopathy, present).
has_suspicious_feature(NoduleId) :-
    nlp_entity(NoduleId, _, pleural_invasion, present).


/* ============================================================
 * SECTION 4: TNM STAGING (AJCC 8th Edition)
 * ============================================================
 * T = Tumor size and local invasion
 * N = Lymph node involvement  
 * M = Metastasis
 * 
 * Source: AJCC Cancer Staging Manual, 8th Edition
 */

% T-Stage based on tumor size (primary tumor)
% Tis: Carcinoma in situ
t_stage(NoduleId, 'Tis', 'Carcinoma in situ') :-
    nlp_entity(NoduleId, _, histology, carcinoma_in_situ).

% T1: Tumor <=3cm
t_stage(NoduleId, 'T1a', 'Tumor <=1cm') :-
    nodule_size(NoduleId, Size),
    Size =< 10.

t_stage(NoduleId, 'T1b', 'Tumor >1-2cm') :-
    nodule_size(NoduleId, Size),
    Size > 10, Size =< 20.

t_stage(NoduleId, 'T1c', 'Tumor >2-3cm') :-
    nodule_size(NoduleId, Size),
    Size > 20, Size =< 30.

% T2: Tumor >3-5cm or involves main bronchus
t_stage(NoduleId, 'T2a', 'Tumor >3-4cm') :-
    nodule_size(NoduleId, Size),
    Size > 30, Size =< 40.

t_stage(NoduleId, 'T2b', 'Tumor >4-5cm') :-
    nodule_size(NoduleId, Size),
    Size > 40, Size =< 50.

% T3: Tumor >5-7cm or chest wall invasion
t_stage(NoduleId, 'T3', 'Tumor >5-7cm') :-
    nodule_size(NoduleId, Size),
    Size > 50, Size =< 70.

t_stage(NoduleId, 'T3', 'Chest wall invasion') :-
    nlp_entity(NoduleId, _, chest_wall_invasion, present).

% T4: Tumor >7cm or invades critical structures
t_stage(NoduleId, 'T4', 'Tumor >7cm') :-
    nodule_size(NoduleId, Size),
    Size > 70.

t_stage(NoduleId, 'T4', 'Mediastinal invasion') :-
    nlp_entity(NoduleId, _, mediastinal_invasion, present).

% N-Stage (lymph node involvement)
n_stage(NoduleId, 'N0', 'No regional lymph node metastasis') :-
    \+ nlp_entity(NoduleId, _, lymph_node, _).

n_stage(NoduleId, 'N1', 'Ipsilateral peribronchial/hilar nodes') :-
    nlp_entity(NoduleId, _, lymph_node, ipsilateral_hilar).

n_stage(NoduleId, 'N2', 'Ipsilateral mediastinal nodes') :-
    nlp_entity(NoduleId, _, lymph_node, ipsilateral_mediastinal).

n_stage(NoduleId, 'N3', 'Contralateral nodes') :-
    nlp_entity(NoduleId, _, lymph_node, contralateral).

% M-Stage (distant metastasis)
m_stage(NoduleId, 'M0', 'No distant metastasis') :-
    \+ nlp_entity(NoduleId, _, metastasis, _).

m_stage(NoduleId, 'M1a', 'Intrathoracic metastasis') :-
    nlp_entity(NoduleId, _, metastasis, intrathoracic).

m_stage(NoduleId, 'M1b', 'Single extrathoracic metastasis') :-
    nlp_entity(NoduleId, _, metastasis, single_extrathoracic).

m_stage(NoduleId, 'M1c', 'Multiple extrathoracic metastases') :-
    nlp_entity(NoduleId, _, metastasis, multiple_extrathoracic).

% Overall stage grouping
stage_group(NoduleId, 'IA1') :-
    t_stage(NoduleId, 'T1a', _),
    n_stage(NoduleId, 'N0', _),
    m_stage(NoduleId, 'M0', _).

stage_group(NoduleId, 'IA2') :-
    t_stage(NoduleId, 'T1b', _),
    n_stage(NoduleId, 'N0', _),
    m_stage(NoduleId, 'M0', _).

stage_group(NoduleId, 'IA3') :-
    t_stage(NoduleId, 'T1c', _),
    n_stage(NoduleId, 'N0', _),
    m_stage(NoduleId, 'M0', _).

stage_group(NoduleId, 'IB') :-
    t_stage(NoduleId, 'T2a', _),
    n_stage(NoduleId, 'N0', _),
    m_stage(NoduleId, 'M0', _).

stage_group(NoduleId, 'IIA') :-
    t_stage(NoduleId, 'T2b', _),
    n_stage(NoduleId, 'N0', _),
    m_stage(NoduleId, 'M0', _).

stage_group(NoduleId, 'IIB') :-
    (t_stage(NoduleId, 'T1a', _) ; t_stage(NoduleId, 'T1b', _) ; t_stage(NoduleId, 'T1c', _) ; t_stage(NoduleId, 'T2a', _) ; t_stage(NoduleId, 'T2b', _)),
    n_stage(NoduleId, 'N1', _),
    m_stage(NoduleId, 'M0', _).

stage_group(NoduleId, 'IIIA') :-
    n_stage(NoduleId, 'N2', _),
    m_stage(NoduleId, 'M0', _).

stage_group(NoduleId, 'IV') :-
    m_stage(NoduleId, Stage, _),
    (Stage == 'M1a' ; Stage == 'M1b' ; Stage == 'M1c').


/* ============================================================
 * SECTION 5: MULTI-AGENT CONSENSUS CALCULATION
 * ============================================================
 * Weighted voting algorithm for combining agent predictions.
 */

% Calculate weighted consensus from all agent findings
% Returns: weighted average probability and confidence
calculate_consensus(NoduleId, WeightedProb, Confidence) :-
    findall(
        prob(Agent, Prob, Weight),
        (
            agent_finding(NoduleId, Agent, Prob, _),
            get_agent_weight(Agent, Weight)
        ),
        Findings
    ),
    Findings \= [],
    sum_weighted_probs(Findings, TotalWeightedProb, TotalWeight),
    TotalWeight > 0,
    WeightedProb is TotalWeightedProb / TotalWeight,
    calculate_agreement(Findings, WeightedProb, Confidence).

% Helper: Sum weighted probabilities
sum_weighted_probs([], 0, 0).
sum_weighted_probs([prob(_, Prob, Weight)|Rest], TotalProb, TotalWeight) :-
    sum_weighted_probs(Rest, RestProb, RestWeight),
    TotalProb is RestProb + (Prob * Weight),
    TotalWeight is RestWeight + Weight.

% Calculate agreement/confidence based on variance
calculate_agreement(Findings, MeanProb, Confidence) :-
    length(Findings, N),
    N > 1,
    findall(
        Diff,
        (
            member(prob(_, Prob, _), Findings),
            Diff is (Prob - MeanProb) ** 2
        ),
        Diffs
    ),
    sum_list(Diffs, SumDiffs),
    Variance is SumDiffs / N,
    StdDev is sqrt(Variance),
    % Confidence decreases as disagreement increases
    Confidence is max(0, 1 - (StdDev * 3)).

calculate_agreement(_, _, 0.8) :- !.  % Default for single agent


/* ============================================================
 * SECTION 6: DISAGREEMENT DETECTION AND RESOLUTION
 * ============================================================
 * Detect when agents disagree significantly and apply resolution.
 */

% Check if there's significant disagreement (std dev > 0.15)
has_disagreement(NoduleId) :-
    findall(
        Prob,
        agent_finding(NoduleId, _, Prob, _),
        Probs
    ),
    length(Probs, N),
    N >= 2,
    mean(Probs, Mean),
    variance(Probs, Mean, Var),
    StdDev is sqrt(Var),
    StdDev > 0.15.

% Mean of a list
mean(List, Mean) :-
    sum_list(List, Sum),
    length(List, N),
    Mean is Sum / N.

% Variance of a list
variance(List, Mean, Var) :-
    findall(D, (member(X, List), D is (X - Mean) ** 2), Diffs),
    sum_list(Diffs, SumDiffs),
    length(List, N),
    Var is SumDiffs / N.

% Disagreement resolution strategies
% 1. Trust CNN radiologists more when NLP agrees with them
resolve_disagreement(NoduleId, FinalProb, Strategy) :-
    has_disagreement(NoduleId),
    cnn_radiologist_consensus(NoduleId, CNNProb),
    pathologist_consensus(NoduleId, PathProb),
    abs(CNNProb - PathProb) < 0.2,
    FinalProb is (CNNProb * 0.6 + PathProb * 0.4),
    Strategy = 'cnn_nlp_agreement'.

% 2. Use rule-based as tiebreaker when CNN radiologists disagree
resolve_disagreement(NoduleId, FinalProb, Strategy) :-
    has_disagreement(NoduleId),
    agent_finding(NoduleId, radiologist_rules, RuleProb, _),
    cnn_radiologist_consensus(NoduleId, CNNProb),
    FinalProb is (CNNProb * 0.5 + RuleProb * 0.5),
    Strategy = 'rule_based_tiebreaker'.

% 3. Default: use weighted average
resolve_disagreement(NoduleId, FinalProb, Strategy) :-
    calculate_consensus(NoduleId, FinalProb, _),
    Strategy = 'weighted_average'.

% CNN radiologist consensus
cnn_radiologist_consensus(NoduleId, AvgProb) :-
    findall(
        Prob,
        (
            agent_finding(NoduleId, Agent, Prob, _),
            agent_type(Agent, radiologist, cnn)
        ),
        Probs
    ),
    Probs \= [],
    mean(Probs, AvgProb).

% Pathologist consensus
pathologist_consensus(NoduleId, AvgProb) :-
    findall(
        Prob,
        (
            agent_finding(NoduleId, Agent, Prob, _),
            is_pathologist(Agent)
        ),
        Probs
    ),
    Probs \= [],
    mean(Probs, AvgProb).


/* ============================================================
 * SECTION 7: CLINICAL RECOMMENDATIONS
 * ============================================================
 * Management recommendations based on Lung-RADS and consensus.
 */

% Get recommendation based on Lung-RADS category
recommendation(NoduleId, Category, Action, Urgency) :-
    lung_rads(NoduleId, Category, _),
    category_recommendation(Category, Action, Urgency).

category_recommendation(1, 'Continue annual screening', low).
category_recommendation(2, 'Continue annual screening', low).
category_recommendation(3, 'Follow-up CT in 6 months', medium).
category_recommendation('4A', 'Follow-up CT in 3 months', high).
category_recommendation('4B', 'PET-CT and/or tissue sampling', high).
category_recommendation('4X', 'Urgent PET-CT and tissue sampling', critical).

% Additional recommendations based on features
additional_recommendation(NoduleId, 'Consider surgical consultation') :-
    stage_group(NoduleId, Stage),
    (Stage == 'IA1' ; Stage == 'IA2' ; Stage == 'IA3' ; Stage == 'IB').

additional_recommendation(NoduleId, 'Multidisciplinary tumor board review') :-
    has_disagreement(NoduleId),
    calculate_consensus(NoduleId, Prob, _),
    Prob > 0.5.


/* ============================================================
 * SECTION 8: FULL ASSESSMENT GENERATION
 * ============================================================
 * Combine all analyses into comprehensive assessment.
 */

% Generate full multi-agent assessment
full_assessment(NoduleId, Assessment) :-
    % Get size and texture
    (nodule_size(NoduleId, Size) -> true ; Size = unknown),
    (nodule_texture(NoduleId, Texture) -> true ; Texture = unknown),
    
    % Get Lung-RADS
    (lung_rads(NoduleId, LungRads, LungRadsDesc) -> true ; 
        (LungRads = unknown, LungRadsDesc = 'Unable to classify')),
    
    % Get TNM if applicable
    (t_stage(NoduleId, TStage, _) -> true ; TStage = 'TX'),
    (n_stage(NoduleId, NStage, _) -> true ; NStage = 'NX'),
    (m_stage(NoduleId, MStage, _) -> true ; MStage = 'MX'),
    
    % Get consensus
    (calculate_consensus(NoduleId, Probability, Confidence) -> true ;
        (Probability = 0.5, Confidence = 0.0)),
    
    % Check for disagreement
    (has_disagreement(NoduleId) -> Disagreement = yes ; Disagreement = no),
    
    % Get recommendation
    (recommendation(NoduleId, _, Action, Urgency) -> true ;
        (Action = 'Clinical correlation recommended', Urgency = medium)),
    
    Assessment = assessment{
        nodule_id: NoduleId,
        size_mm: Size,
        texture: Texture,
        lung_rads_category: LungRads,
        lung_rads_description: LungRadsDesc,
        tnm_stage: tnm(TStage, NStage, MStage),
        consensus_probability: Probability,
        confidence: Confidence,
        agent_disagreement: Disagreement,
        recommendation: Action,
        urgency: Urgency
    }.


/* ============================================================
 * SECTION 9: AGENT FINDING MANAGEMENT
 * ============================================================
 * Utilities for managing dynamic agent findings.
 */

% Add a new agent finding
add_agent_finding(NoduleId, Agent, Probability, Class) :-
    retractall(agent_finding(NoduleId, Agent, _, _)),
    assertz(agent_finding(NoduleId, Agent, Probability, Class)).

% Clear all findings for a nodule
clear_findings(NoduleId) :-
    retractall(agent_finding(NoduleId, _, _, _)),
    retractall(agent_features(NoduleId, _, _)),
    retractall(nlp_entity(NoduleId, _, _, _)),
    retractall(nodule_size(NoduleId, _)),
    retractall(nodule_texture(NoduleId, _)),
    retractall(consensus_result(NoduleId, _, _, _)).

% Get all agent findings for a nodule
get_all_findings(NoduleId, Findings) :-
    findall(
        finding(Agent, Prob, Class),
        agent_finding(NoduleId, Agent, Prob, Class),
        Findings
    ).


/* ============================================================
 * SECTION 10: PROBABILITY TO CLASS MAPPING
 * ============================================================
 */

% Convert probability to malignancy class (1-5)
probability_to_class(Prob, 1) :- Prob < 0.2.
probability_to_class(Prob, 2) :- Prob >= 0.2, Prob < 0.4.
probability_to_class(Prob, 3) :- Prob >= 0.4, Prob < 0.6.
probability_to_class(Prob, 4) :- Prob >= 0.6, Prob < 0.8.
probability_to_class(Prob, 5) :- Prob >= 0.8.

% Convert class to risk level
class_to_risk(1, low).
class_to_risk(2, low).
class_to_risk(3, intermediate).
class_to_risk(4, high).
class_to_risk(5, high).


/* ============================================================
 * SECTION 11: NATURAL LANGUAGE EXPLANATION GENERATION
 * ============================================================
 * Generate human-readable explanations of consensus decisions.
 * 
 * EDUCATIONAL PURPOSE:
 * This demonstrates how Prolog can generate structured natural
 * language explanations for AI decisions, improving interpretability.
 */

% Main explanation generator
% Produces a structured explanation of the consensus decision
generate_explanation(NoduleId, Explanation) :-
    explain_agents(NoduleId, AgentExpl),
    explain_consensus(NoduleId, ConsensusExpl),
    explain_resolution(NoduleId, ResolutionExpl),
    explain_recommendation(NoduleId, RecommendationExpl),
    Explanation = explanation{
        nodule_id: NoduleId,
        agent_summary: AgentExpl,
        consensus_summary: ConsensusExpl,
        resolution_summary: ResolutionExpl,
        recommendation_summary: RecommendationExpl
    }.

% Explain agent findings
explain_agents(NoduleId, Summary) :-
    findall(
        AgentInfo,
        (
            agent_finding(NoduleId, Agent, Prob, Class),
            get_agent_weight(Agent, Weight),
            format(atom(AgentInfo), '~w reported probability ~2f (class ~w, weight ~2f)', 
                   [Agent, Prob, Class, Weight])
        ),
        Summaries
    ),
    atomic_list_concat(Summaries, '; ', Summary).

explain_agents(NoduleId, 'No agent findings available') :-
    \+ agent_finding(NoduleId, _, _, _).

% Explain consensus calculation
explain_consensus(NoduleId, Summary) :-
    calculate_consensus(NoduleId, Prob, Confidence),
    probability_to_class(Prob, Class),
    (has_disagreement(NoduleId) -> 
        Disagreement = 'with significant disagreement' ; 
        Disagreement = 'with good agreement'),
    format(atom(Summary), 
           'Weighted consensus: ~2f probability (class ~w), confidence ~2f ~w',
           [Prob, Class, Confidence, Disagreement]).

explain_consensus(NoduleId, 'Unable to calculate consensus') :-
    \+ calculate_consensus(NoduleId, _, _).

% Explain disagreement resolution
explain_resolution(NoduleId, Summary) :-
    has_disagreement(NoduleId),
    resolve_disagreement(NoduleId, FinalProb, Strategy),
    strategy_description(Strategy, StrategyDesc),
    format(atom(Summary),
           'Disagreement resolved using ~w strategy. Final probability: ~2f',
           [StrategyDesc, FinalProb]).

explain_resolution(NoduleId, 'No disagreement requiring resolution') :-
    \+ has_disagreement(NoduleId).

% Strategy descriptions for human readability
strategy_description('cnn_nlp_agreement', 'CNN-NLP agreement (weighted CNN 60%, NLP 40%)').
strategy_description('rule_based_tiebreaker', 'rule-based tiebreaker (CNN 50%, rules 50%)').
strategy_description('weighted_average', 'weighted average of all agents').

% Explain recommendation
explain_recommendation(NoduleId, Summary) :-
    recommendation(NoduleId, Category, Action, Urgency),
    format(atom(Summary),
           'Lung-RADS ~w: ~w (urgency: ~w)',
           [Category, Action, Urgency]).

explain_recommendation(NoduleId, 'Clinical correlation recommended') :-
    \+ recommendation(NoduleId, _, _, _).

% Generate full natural language explanation (sentences)
generate_narrative(NoduleId, Narrative) :-
    % Get nodule info
    (nodule_size(NoduleId, Size) -> true ; Size = unknown),
    (nodule_texture(NoduleId, Texture) -> true ; Texture = unknown),
    
    % Get agent count
    findall(Agent, agent_finding(NoduleId, Agent, _, _), Agents),
    length(Agents, NumAgents),
    
    % Get consensus
    (calculate_consensus(NoduleId, Prob, Confidence) -> 
        (probability_to_class(Prob, Class),
         class_to_risk(Class, Risk)) ; 
        (Prob = 0.5, Confidence = 0, Class = 3, Risk = intermediate)),
    
    % Build narrative
    format(atom(SentenceOne),
           'This ~w nodule (~w mm) was analyzed by ~w expert agents.',
           [Texture, Size, NumAgents]),
    
    format(atom(SentenceTwo),
           'The consensus malignancy probability is ~2f (class ~w, ~w risk) with ~2f confidence.',
           [Prob, Class, Risk, Confidence]),
    
    % Add disagreement info if applicable
    (has_disagreement(NoduleId) ->
        (resolve_disagreement(NoduleId, _, Strategy),
         strategy_description(Strategy, StrategyDesc),
         format(atom(SentenceThree),
                'Significant disagreement was detected and resolved using ~w.',
                [StrategyDesc])) ;
        SentenceThree = 'All agents showed good agreement.'),
    
    % Add recommendation
    (recommendation(NoduleId, Category, Action, Urgency) ->
        format(atom(SentenceFour),
               'Based on Lung-RADS category ~w, the recommendation is: ~w (~w priority).',
               [Category, Action, Urgency]) ;
        SentenceFour = 'Clinical correlation is recommended.'),
    
    atomic_list_concat([SentenceOne, SentenceTwo, SentenceThree, SentenceFour], ' ', Narrative).

% List which rules fired for a decision
list_fired_rules(NoduleId, FiredRules) :-
    findall(
        Rule,
        (
            (lung_rads(NoduleId, _, _) -> Rule = lung_rads_classification ; fail) ;
            (t_stage(NoduleId, _, _) -> Rule = t_staging ; fail) ;
            (has_disagreement(NoduleId) -> Rule = disagreement_detection ; fail) ;
            (has_suspicious_feature(NoduleId) -> Rule = suspicious_feature_upgrade ; fail)
        ),
        FiredRules
    ).

% Agent certainty integration (for Pathologist-3 context agent)
:- dynamic agent_certainty/3.  % agent_certainty(NoduleId, Agent, Certainty)

% Get certainty-adjusted probability
certainty_adjusted_prob(NoduleId, Agent, AdjustedProb) :-
    agent_finding(NoduleId, Agent, BaseProb, _),
    (agent_certainty(NoduleId, Agent, Certainty) ->
        (Certainty == negated -> AdjustedProb is 0.1 ;
         Certainty == uncertain -> AdjustedProb is 0.5 * BaseProb + 0.25 ;
         AdjustedProb = BaseProb) ;
        AdjustedProb = BaseProb).

% Explain certainty from context agent
explain_certainty(NoduleId, CertaintyExpl) :-
    findall(
        Info,
        (
            agent_certainty(NoduleId, Agent, Certainty),
            format(atom(Info), '~w: ~w', [Agent, Certainty])
        ),
        Infos
    ),
    (Infos \= [] ->
        atomic_list_concat(Infos, ', ', CertaintyExpl) ;
        CertaintyExpl = 'No certainty information available').
