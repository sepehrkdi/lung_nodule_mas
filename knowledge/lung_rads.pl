/**
 * Lung nodule classification rules based on Lung-RADS and Fleischner guidelines.
 */


/* ============================================================
 * SECTION 1: SEMANTIC MAPPINGS
 * ============================================================
 */

% Texture mappings (categorical)
texture_label(1, 'non_solid_ggo').
texture_label(2, 'non_solid_mixed').
texture_label(3, 'part_solid').
texture_label(4, 'mostly_solid').
texture_label(5, 'solid').

% Margin mappings (lower = more concerning)
margin_label(1, 'poorly_defined').
margin_label(2, 'near_poorly_defined').
margin_label(3, 'medium').
margin_label(4, 'near_sharp').
margin_label(5, 'sharp').

% Spiculation mappings (higher = more concerning)
spiculation_label(1, 'none').
spiculation_label(2, 'nearly_none').
spiculation_label(3, 'medium').
spiculation_label(4, 'near_marked').
spiculation_label(5, 'marked').

% Binary malignancy classification (NLMCXR)
% 0 = benign, 1 = malignant
malignancy_label(0, 'benign').
malignancy_label(1, 'malignant').

% Calcification patterns
calcification_label(1, 'popcorn').      % Benign
calcification_label(2, 'laminated').    % Benign
calcification_label(3, 'solid').        % Suspicious
calcification_label(4, 'non_central').  % Suspicious
calcification_label(5, 'central').      % Benign
calcification_label(6, 'absent').       % Neutral


/* ============================================================
 * SECTION 2: CHARACTERISTIC ASSESSMENT RULES
 * ============================================================
 * These rules derive risk factors from nodule characteristics.
 * DEMONSTRATES: Rule chaining, logical inference
 */

% --- Size-based risk assessment ---
% EDUCATIONAL NOTE: Prolog uses pattern matching (unification)
% The variable S unifies with the size value from facts

small_nodule(N) :-
    size(N, S),
    S < 6.

medium_nodule(N) :-
    size(N, S),
    S >= 6,
    S < 15.

large_nodule(N) :-
    size(N, S),
    S >= 15.

very_large_nodule(N) :-
    size(N, S),
    S >= 30.


% --- Texture-based risk ---
% Ground-glass and part-solid nodules have different thresholds

is_solid(N) :-
    texture(N, T),
    T >= 4.

is_part_solid(N) :-
    texture(N, T),
    T == 3.

is_ground_glass(N) :-
    texture(N, T),
    T =< 2.


% --- Morphology-based suspicion ---
% EDUCATIONAL NOTE: These rules demonstrate FOL inference
% A nodule is suspicious if it has concerning features

has_spiculation(N) :-
    spiculation(N, S),
    S >= 3.

has_marked_spiculation(N) :-
    spiculation(N, S),
    S >= 4.

has_poor_margins(N) :-
    margin(N, M),
    M =< 2.

has_lobulation(N) :-
    lobulation(N, L),
    L >= 3.


% --- Benign calcification patterns ---
% EDUCATIONAL NOTE: Negation as Failure (\+)
% Benign if calcification is popcorn, laminated, or central

has_benign_calcification(N) :-
    calcification(N, C),
    (C == 1 ; C == 2 ; C == 5).  % Popcorn, Laminated, or Central

has_suspicious_calcification(N) :-
    calcification(N, C),
    (C == 3 ; C == 4).  % Solid or Non-central


/* ============================================================
 * SECTION 3: RISK STRATIFICATION RULES
 * ============================================================
 * Combines individual features into overall risk assessment.
 * DEMONSTRATES: Rule composition, multiple conditions
 */

% --- High risk features (any of these increases suspicion) ---
high_risk_morphology(N) :-
    has_marked_spiculation(N),
    has_poor_margins(N).

high_risk_morphology(N) :-
    very_large_nodule(N).

% --- Overall risk level ---
% EDUCATIONAL NOTE: Multiple rules for same predicate = OR

risk_level(N, low) :-
    small_nodule(N),
    \+ has_spiculation(N),
    \+ has_poor_margins(N).

risk_level(N, low) :-
    has_benign_calcification(N).

risk_level(N, intermediate) :-
    medium_nodule(N),
    \+ high_risk_morphology(N).

risk_level(N, intermediate) :-
    is_part_solid(N),
    \+ large_nodule(N).

risk_level(N, high) :-
    high_risk_morphology(N).

risk_level(N, high) :-
    large_nodule(N),
    is_solid(N),
    \+ has_benign_calcification(N).

% Default to intermediate if no other rule applies
risk_level(N, intermediate) :-
    nodule(N),
    \+ risk_level(N, low),
    \+ risk_level(N, high).


/* ============================================================
 * SECTION 4: LUNG-RADS CLASSIFICATION
 * ============================================================
 * Lung-RADS (Lung CT Screening Reporting and Data System)
 * 
 * Categories:
 * 1 - Negative (no nodules)
 * 2 - Benign appearance or behavior
 * 3 - Probably benign
 * 4A - Suspicious (short-term follow-up)
 * 4B - Suspicious (tissue sampling)
 * 4X - Additional features increase suspicion
 * 
 * DEMONSTRATES: Complex rule logic, category assignment
 */

% Category 2: Benign nodules
lung_rads_category(N, 2) :-
    is_solid(N),
    size(N, S),
    S < 6.

lung_rads_category(N, 2) :-
    is_part_solid(N),
    size(N, S),
    S < 6.

lung_rads_category(N, 2) :-
    has_benign_calcification(N).

lung_rads_category(N, 2) :-
    is_ground_glass(N),
    size(N, S),
    S < 30.

% Category 3: Probably benign
lung_rads_category(N, 3) :-
    is_solid(N),
    size(N, S),
    S >= 6,
    S < 8,
    \+ has_spiculation(N).

lung_rads_category(N, 3) :-
    is_part_solid(N),
    size(N, S),
    S >= 6,
    S < 8.

% Category 4A: Suspicious - needs short-term follow-up
lung_rads_category(N, '4A') :-
    is_solid(N),
    size(N, S),
    S >= 8,
    S < 15,
    \+ high_risk_morphology(N).

lung_rads_category(N, '4A') :-
    is_part_solid(N),
    size(N, S),
    S >= 8.

% Category 4B: Very suspicious - tissue sampling recommended
lung_rads_category(N, '4B') :-
    is_solid(N),
    size(N, S),
    S >= 15.

lung_rads_category(N, '4B') :-
    high_risk_morphology(N),
    size(N, S),
    S >= 8.

% Category 4X: Additional features increase suspicion
% EDUCATIONAL NOTE: This shows how we can add modifiers
lung_rads_category(N, '4X') :-
    lung_rads_category(N, Cat),
    (Cat == '4A' ; Cat == '4B'),
    (has_marked_spiculation(N) ; has_suspicious_calcification(N)).


/* ============================================================
 * SECTION 5: CLINICAL RECOMMENDATIONS
 * ============================================================
 * Treatment/follow-up recommendations based on classification.
 * DEMONSTRATES: Decision rules, action derivation
 */

% Recommendation rules based on Lung-RADS
recommendation(N, 'annual_screening') :-
    lung_rads_category(N, 2).

recommendation(N, 'followup_6_months') :-
    lung_rads_category(N, 3).

recommendation(N, 'followup_3_months') :-
    lung_rads_category(N, '4A').

recommendation(N, 'pet_ct_or_biopsy') :-
    lung_rads_category(N, '4B').

recommendation(N, 'pet_ct_or_biopsy') :-
    lung_rads_category(N, '4X').

% Additional recommendations based on specific features
recommendation(N, 'consider_biopsy') :-
    has_marked_spiculation(N),
    large_nodule(N).

recommendation(N, 'growth_assessment') :-
    is_part_solid(N),
    medium_nodule(N).


/* ============================================================
 * SECTION 6: TNM STAGING (SIMPLIFIED)
 * ============================================================
 * Tumor-Node-Metastasis staging for lung cancer.
 * This is simplified for educational purposes.
 * 
 * DEMONSTRATES: Multi-factor classification, staging logic
 */

% T stage based on size (simplified)
t_stage(N, 't1a') :- size(N, S), S =< 10.
t_stage(N, 't1b') :- size(N, S), S > 10, S =< 20.
t_stage(N, 't1c') :- size(N, S), S > 20, S =< 30.
t_stage(N, 't2a') :- size(N, S), S > 30, S =< 40.
t_stage(N, 't2b') :- size(N, S), S > 40, S =< 50.
t_stage(N, 't3')  :- size(N, S), S > 50, S =< 70.
t_stage(N, 't4')  :- size(N, S), S > 70.

% N stage (lymph node involvement) - would come from pathology
n_stage(N, 'n0') :- \+ lymph_involvement(N, _).
n_stage(N, 'n1') :- lymph_involvement(N, ipsilateral_hilar).
n_stage(N, 'n2') :- lymph_involvement(N, ipsilateral_mediastinal).
n_stage(N, 'n3') :- lymph_involvement(N, contralateral).

% M stage (metastasis) - would come from full body imaging
m_stage(N, 'm0') :- \+ metastasis(N, _).
m_stage(N, 'm1a') :- metastasis(N, intrathoracic).
m_stage(N, 'm1b') :- metastasis(N, single_extrathoracic).
m_stage(N, 'm1c') :- metastasis(N, multiple_extrathoracic).


/* ============================================================
 * SECTION 7: UTILITY PREDICATES
 * ============================================================
 * Helper predicates for querying and aggregation.
 * DEMONSTRATES: findall/3, meta-predicates
 */

% Collect all recommendations for a nodule
% EDUCATIONAL NOTE: findall/3 demonstrates backtracking
all_recommendations(N, Recs) :-
    findall(R, recommendation(N, R), Recs).

% Get Lung-RADS-only assessment (see multi_agent_consensus.pl for full_assessment/2)
lung_rads_assessment(N, Assessment) :-
    nodule(N),
    (lung_rads_category(N, LungRads) -> true ; LungRads = unknown),
    (risk_level(N, Risk) -> true ; Risk = unknown),
    all_recommendations(N, Recs),
    Assessment = assessment(
        nodule_id(N),
        lung_rads(LungRads),
        risk(Risk),
        recommendations(Recs)
    ).

% Check if nodule needs urgent attention
needs_urgent_attention(N) :-
    lung_rads_category(N, Cat),
    (Cat == '4B' ; Cat == '4X').

% Summary predicate for malignancy prediction
% Maps Lung-RADS to binary benign/malignant
likely_benign(N) :-
    lung_rads_category(N, Cat),
    (Cat == 2 ; Cat == 3).

likely_malignant(N) :-
    lung_rads_category(N, Cat),
    (Cat == '4A' ; Cat == '4B' ; Cat == '4X').


/* ============================================================
 * SECTION 8: DYNAMIC PREDICATES
 * ============================================================
 * These predicates are asserted dynamically by the Python agents.
 * We declare them as dynamic so Prolog knows they can be modified.
 */

:- dynamic nodule/1.
:- dynamic size/2.
:- dynamic texture/2.
:- dynamic margin/2.
:- dynamic spiculation/2.
:- dynamic lobulation/2.
:- dynamic calcification/2.
:- dynamic sphericity/2.
:- dynamic malignancy/2.
:- dynamic lymph_involvement/2.
:- dynamic metastasis/2.

% Also dynamic for agent findings
:- dynamic image_probability/2.
:- dynamic nlp_finding/2.


/* ============================================================
 * SECTION 9: EXAMPLE QUERIES (for testing)
 * ============================================================
 * Uncomment and run in Prolog interpreter to test:
 * 
 * ?- assertz(nodule(n001)).
 * ?- assertz(size(n001, 18)).
 * ?- assertz(texture(n001, 5)).
 * ?- assertz(spiculation(n001, 4)).
 * ?- assertz(margin(n001, 2)).
 * ?- assertz(calcification(n001, 6)).
 * 
 * ?- lung_rads_category(n001, Cat).
 * Cat = '4B'.
 * 
 * ?- all_recommendations(n001, Recs).
 * Recs = [pet_ct_or_biopsy, consider_biopsy].
 * 
 * ?- lung_rads_assessment(n001, A).
 * A = assessment(nodule_id(n001), lung_rads('4B'), risk(high), 
 *                recommendations([pet_ct_or_biopsy, consider_biopsy])).
 */
