// Oncologist Agent Plans (AgentSpeak for SPADE-BDI)
// Synthesizes findings using Prolog reasoning.

// --------------------------------------------
// Initial beliefs
// --------------------------------------------
prolog_loaded(false).
knowledge_base("lung_rads.pl").
waiting_for(radiologist).
waiting_for(pathologist).

// --------------------------------------------
// Plan: Initialize Prolog knowledge base
// --------------------------------------------
+!initialize : prolog_loaded(false)
    <- .print("Loading Prolog knowledge base...");
       .load_prolog_kb;
       -prolog_loaded(false);
       +prolog_loaded(true);
       .print("Prolog KB ready").

// --------------------------------------------
// Reactive Plan: Receive radiologist findings
// --------------------------------------------
// Triggered when radiologist sends findings via tell
//
+radiologist_findings(NoduleId, Probability, Class, Size)[source(radiologist)]
    <- .print("Received radiologist findings for: ", NoduleId);
       +has_radiologist_findings(NoduleId);
       +rad_probability(NoduleId, Probability)[source(radiologist)];
       +rad_class(NoduleId, Class)[source(radiologist)];
       +rad_size(NoduleId, Size)[source(radiologist)];
       -waiting_for(radiologist);
       !check_ready(NoduleId).

// --------------------------------------------
// Reactive Plan: Receive pathologist findings
// --------------------------------------------
+pathologist_findings(NoduleId, Size, Texture, Margin, Spiculation)[source(pathologist)]
    <- .print("Received pathologist findings for: ", NoduleId);
       +has_pathologist_findings(NoduleId);
       +path_size(NoduleId, Size)[source(pathologist)];
       +path_texture(NoduleId, Texture)[source(pathologist)];
       +path_margin(NoduleId, Margin)[source(pathologist)];
       +path_spiculation(NoduleId, Spiculation)[source(pathologist)];
       -waiting_for(pathologist);
       !check_ready(NoduleId).

// --------------------------------------------
// Plan: Check if ready to assess
// --------------------------------------------
+!check_ready(NoduleId) : has_radiologist_findings(NoduleId) & 
                          has_pathologist_findings(NoduleId)
    <- .print("All findings received, starting assessment");
       !assess(NoduleId).

+!check_ready(NoduleId) : not has_radiologist_findings(NoduleId)
    <- .print("Waiting for radiologist findings...").

+!check_ready(NoduleId) : not has_pathologist_findings(NoduleId)
    <- .print("Waiting for pathologist findings...").

// --------------------------------------------
// Plan: Full assessment using Prolog
// --------------------------------------------
+!assess(NoduleId) : prolog_loaded(true)
    <- .print("Assessing nodule: ", NoduleId);
       !merge_findings(NoduleId);
       !apply_lung_rads(NoduleId);
       !determine_staging(NoduleId);
       !generate_recommendation(NoduleId);
       !finalize(NoduleId).

+!assess(NoduleId) : prolog_loaded(false)
    <- !initialize;
       !assess(NoduleId).

// --------------------------------------------
// Plan: Merge findings from both sources
// --------------------------------------------
// EDUCATIONAL: Conflict resolution when sources disagree
//
+!merge_findings(NoduleId) : rad_size(NoduleId, RadSize) & 
                             path_size(NoduleId, PathSize)
    <- .print("Merging findings...");
       .resolve_size(RadSize, PathSize, FinalSize);
       +final_size(NoduleId, FinalSize)[source(self)];
       ?path_texture(NoduleId, Texture);
       +final_texture(NoduleId, Texture)[source(self)];
       ?path_margin(NoduleId, Margin);
       +final_margin(NoduleId, Margin)[source(self)];
       ?path_spiculation(NoduleId, Spic);
       +final_spiculation(NoduleId, Spic)[source(self)].

// --------------------------------------------
// Plan: Apply Lung-RADS using Prolog
// --------------------------------------------
+!apply_lung_rads(NoduleId) : final_size(NoduleId, Size) & 
                              final_texture(NoduleId, Texture)
    <- .print("Querying Prolog for Lung-RADS...");
       .prolog_lung_rads(Size, Texture, Category);
       +lung_rads(NoduleId, Category)[source(self)];
       .print("Lung-RADS category: ", Category).

// --------------------------------------------
// Plan: Determine TNM staging using Prolog
// --------------------------------------------
+!determine_staging(NoduleId) : final_size(NoduleId, Size)
    <- .print("Querying Prolog for TNM staging...");
       .prolog_tnm_stage(Size, TStage);
       +t_stage(NoduleId, TStage)[source(self)];
       .print("T-stage: ", TStage).

// --------------------------------------------
// Plan: Generate recommendation using Prolog
// --------------------------------------------
+!generate_recommendation(NoduleId) : lung_rads(NoduleId, Category)
    <- .print("Querying Prolog for recommendation...");
       .prolog_recommendation(Category, Recommendation);
       +recommendation(NoduleId, Recommendation)[source(self)];
       .print("Recommendation: ", Recommendation).

// --------------------------------------------
// Plan: Finalize and broadcast result
// --------------------------------------------
+!finalize(NoduleId) : lung_rads(NoduleId, Category) & 
                       t_stage(NoduleId, TStage) & 
                       recommendation(NoduleId, Rec)
    <- .print("Assessment complete for: ", NoduleId);
       +assessment_complete(NoduleId)[source(self)];
       // Compute final malignancy probability
       ?rad_probability(NoduleId, Prob);
       +final_probability(NoduleId, Prob)[source(self)];
       // Broadcast to all agents
       .broadcast(tell, final_assessment(NoduleId, Category, TStage, Rec, Prob)).

// --------------------------------------------
// Plan: Handle direct assessment request
// --------------------------------------------
+request(assess, NoduleId, RadFindings, PathFindings)[source(Sender)]
    <- .print("Direct assessment request from: ", Sender);
       !process_direct_request(NoduleId, RadFindings, PathFindings).

+!process_direct_request(NoduleId, RadFindings, PathFindings) : true
    <- // Unpack findings
       .unpack_rad_findings(RadFindings, Prob, Class, Size);
       +rad_probability(NoduleId, Prob);
       +rad_class(NoduleId, Class);
       +rad_size(NoduleId, Size);
       +has_radiologist_findings(NoduleId);
       .unpack_path_findings(PathFindings, PSize, Texture, Margin, Spic);
       +path_size(NoduleId, PSize);
       +path_texture(NoduleId, Texture);
       +path_margin(NoduleId, Margin);
       +path_spiculation(NoduleId, Spic);
       +has_pathologist_findings(NoduleId);
       !assess(NoduleId).

// --------------------------------------------
// Plan: Query for assessment result
// --------------------------------------------
+?get_assessment(NoduleId, Result) : assessment_complete(NoduleId) & 
                                     lung_rads(NoduleId, Cat) & 
                                     t_stage(NoduleId, T) & 
                                     recommendation(NoduleId, Rec) &
                                     final_probability(NoduleId, Prob)
    <- Result = assessment(NoduleId, Cat, T, Rec, Prob).

+?get_assessment(NoduleId, Result) : not assessment_complete(NoduleId)
    <- Result = pending(NoduleId).

// ============================================
// Internal Actions (implemented in Python):
// ============================================
// .load_prolog_kb          - Load lung_rads.pl into PySwip
// .resolve_size(R, P, F)   - Resolve size conflict between sources
// .prolog_lung_rads(Size, Texture, Cat) - Query Lung-RADS category
// .prolog_tnm_stage(Size, TStage)       - Query TNM T-stage
// .prolog_recommendation(Cat, Rec)       - Query recommendation
// .unpack_rad_findings(F, Prob, Class, Size)
// .unpack_path_findings(F, Size, Tex, Mar, Spic)
// ============================================
